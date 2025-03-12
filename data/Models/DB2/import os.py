import os
import csv
import copy
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel
)

from imblearn.over_sampling import ADASYN
from collections import Counter
import wandb
import optuna

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, 
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score,
    roc_curve, precision_recall_curve
)

from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

class ModelStateManager:
    """Manages model state persistence and loading"""
    def __init__(self, run_name: str, output_dir: str):
        self.run_name = run_name
        self.output_dir = output_dir
        self.best_model_dir = os.path.join(output_dir, f"{run_name}_Best_Model")
        self.best_params_file = os.path.join(self.best_model_dir, f"{run_name}_best_params.json")
        
    def save_best_model(self, trainer: transformers.Trainer, params: dict):
        os.makedirs(self.best_model_dir, exist_ok=True)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer, self.best_model_dir)
        with open(self.best_params_file, "w") as f:
            json.dump(params, f)
            
    def load_best_model(self, model_args, num_labels: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.best_model_dir,
            num_labels=num_labels,
            trust_remote_code=True
        ).to(device)
        
        if model_args.use_lora:
            model = PeftModel.from_pretrained(model, self.best_model_dir)
            
        with open(self.best_params_file, "r") as f:
            params = json.load(f)
            
        return model, params

class EvaluationManager:
    """Handles model evaluation and results logging"""
    def __init__(self, state_manager: ModelStateManager):
        self.state = state_manager
        
    def evaluate_model(self, trainer, test_dataset):
        """Run comprehensive evaluation"""
        results = trainer.evaluate(test_dataset)
        
        with open(os.path.join(self.state.output_dir, "final_test_results.json"), "w") as f:
            json.dump(results, f)
            
        plot_confusion_matrix(trainer, test_dataset, 
                            self.state.output_dir, self.state.run_name)
        
        return results

[... CONTINUE WITH ALL EXISTING DATACLASS DEFINITIONS ...]

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        trainer.model.config.save_pretrained(output_dir)
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        if hasattr(trainer.model, 'peft_config'):
            trainer.model.save_pretrained(output_dir)

[... CONTINUE WITH ALL EXISTING UTILITY FUNCTIONS ...]

def build_and_train_model(
    model_args,
    data_args,
    training_args,
    tokenizer,
    train_dataset,
    val_dataset,
    data_collator,
    trial_params: dict = None
):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )

    if model_args.use_lora:
        try:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=list(model_args.lora_target_modules.split(",")),
                lora_dropout=0.2,
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
        except Exception as e:
            logger.warning(f"Failed to apply LoRA config: {e}")
            return None

    if model_args.use_class_weights:
        alpha = (torch.tensor([trial_params["alpha_class_0"], 
                             trial_params["alpha_class_1"]])
                if trial_params else
                compute_class_weights(train_dataset.labels, train_dataset.num_labels))
        gamma = trial_params.get("gamma", 2.0) if trial_params else 2.0
        loss_fn = FocalLoss(alpha=alpha.to(model.device), gamma=gamma)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    local_training_args = copy.deepcopy(training_args)
    if trial_params:
        local_training_args.learning_rate = trial_params["learning_rate"]
        local_training_args.weight_decay = trial_params["weight_decay"]
        local_training_args.num_train_epochs = trial_params["num_train_epochs"]
        local_training_args.per_device_train_batch_size = trial_params["batch_size"]

    trainer_class = CustomTrainer if model_args.use_class_weights else transformers.Trainer
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=local_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=extended_compute_metrics,
        loss_fn=loss_fn if model_args.use_class_weights else None,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.001
            ),
            ProgressCallback()
        ]
    )

    trainer.train()
    return trainer

def create_objective(
    model_args,
    data_args,
    training_args,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    data_collator,
    state_manager: ModelStateManager
):
    def objective(trial):
        trial_params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 9e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 5e-5, 1.5e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 8, 12),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
            "gamma": trial.suggest_float("gamma", 4, 5.5),
            "alpha_class_0": trial.suggest_float("alpha_class_0", 0.1, 0.4),
            "alpha_class_1": trial.suggest_float("alpha_class_1", 0.45, 2)
        }
        
        trainer = build_and_train_model(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            trial_params=trial_params
        )
        
        if trainer is None:
            return float("-inf")
            
        metrics = trainer.state.best_metric_dict
        # combined_score = 0.3 * metrics.get("eval_macro_f1") + 0.4 * metrics.get("eval_auc") + metrics.get("eval_class_1_f1") * 0.3
        # combined_score = 0.1 * metrics.get("eval_macro_f1") + 0.5 * metrics.get("eval_auc") + metrics.get("eval_class_1_f1") * 0.4
        combined_score = 0.4 * metrics.get("eval_auc", 0) + 0.6 * metrics.get("class_1_f1", 0)
        
        if trial.number == 0 or combined_score > trial.study.best_value:
            state_manager.save_best_model(trainer, trial_params)
            
        return combined_score
        
    return objective

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{training_args.output_dir}/training.log"),
            logging.StreamHandler()
        ]
    )
    logger.info("Starting training...")

    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(data_args.data_path)))
    results_dir = Path("./results") / f"DB2_{dataset_name}_Alt5"
    results_dir.mkdir(parents=True, exist_ok=True)

    training_args.output_dir = str(results_dir / "model_output")
    training_args.logging_dir = str(results_dir / "logs")
    
    if model_args.use_wandb:
        run_name = f"DB2_{dataset_name}_Alt5"
        wandb.init(entity='mtamargo', project="ASPECT2", name=f"{run_name}")
        wandb.config.update({**vars(model_args), **vars(data_args), **vars(training_args)})
    else:
        run_name = "no_wandb_run"

    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)

    tokenizer = get_latest_checkpoint_tokenizer(training_args.output_dir)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "train.csv"),
        kmer=data_args.kmer,
    )

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer,
    )
    
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    state_manager = ModelStateManager(
        run_name=f"DB2_{dataset_name}_Alt5",
        output_dir=str(results_dir)
    )
    
    try:
        if model_args.use_optuna:
            study = optuna.create_study(direction="maximize")
            objective_fn = create_objective(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                data_collator=data_collator,
                state_manager=state_manager
            )
            
            study.optimize(objective_fn, n_trials=training_args.optuna_trials)
            
            model, best_params = state_manager.load_best_model(
                model_args=model_args,
                num_labels=train_dataset.num_labels
            )
            
            eval_trainer = (CustomTrainer if model_args.use_class_weights else transformers.Trainer)(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=extended_compute_metrics,
                loss_fn=FocalLoss(
                    alpha=torch.tensor([best_params["alpha_class_0"], 
                                      best_params["alpha_class_1"]]).to(model.device),
                    gamma=best_params["gamma"]
                ) if model_args.use_class_weights else None
            )
            
            evaluator = EvaluationManager(state_manager)
            final_results = evaluator.evaluate_model(eval_trainer, test_dataset)
            logger.info("\n====== Final Trial Summary Report ======")
            logger.info(f"Final Test Results: {final_results}")
            logger.info(f"Optuna finished. Best params: {study.best_params}")
            
            if model_args.use_wandb:
                wandb.log({
                    "best_optuna_params": best_params,
                    "final_test_results": final_results
                })
        else:
            trainer = build_and_train_model(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                data_collator=data_collator
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually (Ctrl+C). Stopping run...")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    finally:
        if model_args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    train()