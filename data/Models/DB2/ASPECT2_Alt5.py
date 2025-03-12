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

# LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# ADASYN (not used in final dataset processing but imported)
from imblearn.over_sampling import ADASYN
from collections import Counter

# WandB
import wandb

# Optuna
import optuna

# Metrics
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, f1_score,
                             roc_curve, precision_recall_curve)

# For ROC/PR/Calibration
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

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M") 
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})

    # Additional flags
    use_wandb: bool = field(default=False, metadata={"help": "Whether to log to Weights & Biases"})
    use_optuna: bool = field(default=False, metadata={"help": "Whether to run hyperparameter search with Optuna"})
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to apply class weights"})
    use_oversampled: bool = field(default=False, metadata={"help": "Whether to use ADASYN oversampled dataset"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output")
    logging_dir: str = field(default="./logs")
    cache_dir: Optional[str] = field(default=None)
    report_to: str = field(default="wandb")
    run_name: str = field(default="DB2")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: int = field(default=10)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=500)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    warmup_ratio: int = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    max_grad_norm: int = field(default=1.0)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=3e-5)
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_macro_f1")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    seed: int = field(default=42)
    optuna_trials: int = field(default=1, metadata={"help": "Number of trials for Optuna HPO"})



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        trainer.model.config.save_pretrained(output_dir)
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        if hasattr(trainer.model, 'peft_config'):
            trainer.model.save_pretrained(output_dir)


def get_alter_of_dna_sequence(sequence: str):
    """Get the reversed complement of the original DNA sequence."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        kmer: int = -1,
    ):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            try:
                texts = [d[0] for d in data]
                labels = [int(d[1]) for d in data]
            except ValueError:
                try:
                    texts = [d[1] for d in data]
                    labels = [int(d[0]) for d in data]
                except ValueError:
                    logging.error(f"Skipping invalid data row: {d}")
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        if kmer != -1:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            if isinstance(texts[0], list):
                # For pair classification, generate k-mer for each piece
                texts = [
                    " ".join([load_or_generate_kmer(data_path, [t], kmer)[0] for t in pair]) 
                    for pair in texts
                ]
            else:
                texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "attention_mask", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


def get_latest_checkpoint_tokenizer(base_dir):
    """Load tokenizer from the latest checkpoint if available."""
    if not Path(base_dir).is_dir():
        print(f"Base directory does not exist: {base_dir}")
        return None
    
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {base_dir}.")
        return None
    
    checkpoint_dir = checkpoint_dirs[0]
    tokenizer_path = os.path.join(base_dir, checkpoint_dir)

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    else:
        print(f"Tokenizer not found in {tokenizer_path}.")
        return None


def softmax(logits: np.ndarray):
    """Softmax for numpy array of shape [batch_size, num_classes]."""
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.detach().cpu()  # Return tensor, not numpy


def extended_compute_metrics(eval_pred):
    """
    Extended compute_metrics with ROC, PR, calibration, plus standard F1, precision, recall, etc.
    """
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    pred_class = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, pred_class)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="binary", pos_label=1, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
    labels, pred_class, average="micro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="weighted", zero_division=0
    )
    per_class_metrics = precision_recall_fscore_support(labels, pred_class, average=None, zero_division=0)

    num_classes = len(np.unique(labels))
    auc_value, pr_auc_value = float("nan"), float("nan")

    if num_classes == 2:
        probs_class1 = softmax(predictions)[:, 1]
        auc_value = roc_auc_score(labels, probs_class1)
        pr_auc_value = average_precision_score(labels, probs_class1)

    class_report = classification_report(
        labels, pred_class, target_names=["Constitutive", "Alternative 5"], output_dict=True
    )

    class_1_f1 = class_report["Alternative 5"]["f1-score"]
    majority_f1 = class_report["Constitutive"]["f1-score"]

    return {
        "accuracy": acc,
    
        "auc": auc_value,
        "pr_auc": pr_auc_value,
        "class_1_f1": class_1_f1,
        "majority_f1": majority_f1,
        
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,
        
        "class_report": class_report
    }


def plot_confusion_matrix(trainer, eval_dataset, results_dir, runname_label):
    """Simple confusion matrix plotting."""
    results_dir = Path(results_dir)
    
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Constitutive", "Alternative 5"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {runname_label}")
    
    plt.savefig(results_dir / f"confusion_matrix_{runname_label}.png")
    plt.close()


class FocalLoss(torch.nn.Module):
    """Focal Loss for classification."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')

    def forward(self, logits, labels):
        # Always move alpha to the same device as logits
        if self.alpha is not None:
            # Update the weight in ce_loss (i.e., class weights)
            self.ce_loss.weight = self.alpha.to(logits.device)
            
        ce_loss = self.ce_loss(logits, labels)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    weights = torch.tensor(class_weights, dtype=torch.float32)
    return weights


class CustomTrainer(transformers.Trainer):
    """
    Custom Trainer to use Focal Loss instead of CrossEntropyLoss.
    """
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ProgressCallback(transformers.TrainerCallback):
    """
    Logs training progress (loss, learning rate) at each logging step.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            if logs is not None and "loss" in logs:
                lr = logs.get('learning_rate', 'N/A')
                step_loss = logs['loss']
                logging.info(f"Step {state.global_step}: loss={step_loss:.4f}, lr={lr}")
   
        


def train():
    
############################################
######### Setup and initialization #########
############################################

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
    
    
####################################################
######### Trainer Setup and Train Function #########
####################################################

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
        """Helper for normal or Optuna-based training loops."""
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
                    target_modules=["Wqkv"],
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)
            except Exception as e:
                logging.warning(f"Failed to apply user-defined target_modules: {e}")
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=list(model_args.lora_target_modules.split(",")),
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)

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
        
        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "args": local_training_args,
            "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
            "train_dataset": train_dataset,
            "eval_dataset": val_dataset,
            "data_collator": data_collator,
            "compute_metrics": extended_compute_metrics,
            "callbacks": [
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=2,
                    early_stopping_threshold=0.001
                ),
                ProgressCallback()
            ]
        }

        if model_args.use_class_weights:
            kwargs["loss_fn"] = loss_fn

        trainer_class = CustomTrainer if model_args.use_class_weights else transformers.Trainer
        trainer = trainer_class(**kwargs)

        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
        return trainer
        
        
##########################################
######### Optuna Helper Function #########
##########################################
        
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
                "num_train_epochs": trial.suggest_int("num_train_epochs", 8, 12),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                
                "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),

                "alpha_class_0": trial.suggest_float("alpha_class_0", 0.05, 0.4),
                "alpha_class_1": trial.suggest_float("alpha_class_1", 1.0, 10.0),
                "gamma": trial.suggest_float("gamma", 1.0, 5.0)
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
                
            metrics = trainer.evaluate()
            # combined_score = 0.3 * metrics.get("eval_macro_f1") + 0.4 * metrics.get("eval_auc") + metrics.get("eval_class_1_f1") * 0.3
            combined_score = 0.1 * metrics.get("eval_macro_f1") + 0.5 * metrics.get("eval_auc") + metrics.get("eval_class_1_f1") * 0.4
            # combined_score = 0.4 * metrics.get("eval_auc", 0) + 0.6 * metrics.get("eval_class_1_f1", 0)
            # combined_score = 0.4 * metrics.get("eval_auc", 0) + 0.6 * metrics.get("eval_macro_f1", 0)
            
            if trial.number == 0 or combined_score > trial.study.best_value:
                state_manager.save_best_model(trainer, trial_params)
                
            return combined_score
            
        return objective
    
########################################
######### Main execution logic #########
########################################

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
            
            
            # ---------- Optional: Log to WandB ----------
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

        # ---------- Optional: Log to WandB ----------
        if model_args.use_wandb:
            wandb.log({
                "best_optuna_params": best_params,
                "final_test_results": final_results
            })

        logger.info("Evaluation completed successfully.")

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually (Ctrl+C). Stopping run...")

    except Exception as e:
        logger.error(f"Error during training: {e}")

    finally:
        if model_args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    train()
