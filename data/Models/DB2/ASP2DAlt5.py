import os
import csv
import copy
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

# Environment settings
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# LoRA and PEFT
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
)

# ADASYN and collections
from imblearn.over_sampling import ADASYN
from collections import Counter

# WandB
import wandb

# Optuna for HPO
import optuna

# Metrics and plotting
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, f1_score,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

#########################################
#           ARGUMENT CLASSES            #
#########################################

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
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine_with_restarts")
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=3e-5)
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    eval_and_save_results: bool = field(default=False)
    save_model: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_macro_f1")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    
    seed: int = field(default=42)
    # For Optuna
    optuna_trials: int = field(default=1, metadata={"help": "Number of trials for Optuna HPO"})

#########################################
#         UTILITY FUNCTIONS             #
#########################################

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dumps to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # Save model configuration
        trainer.model.config.save_pretrained(output_dir)
        # Save model weights (on CPU)
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        if hasattr(trainer.model, 'peft_config'):
            trainer.model.save_pretrained(output_dir)

# Manage saving and loading of the best model.
class ModelStateManager:
    """Manages model state persistence and loading."""
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

# DNA sequence processing helpers
def get_alter_of_dna_sequence(sequence: str):
    """Return the reversed complement of a DNA sequence."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([MAP[c] for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate a k-mer string from a DNA sequence."""
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load precomputed k-mer strings or generate them if not present."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning("Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer

#########################################
#       DATASET AND COLLATION           #
#########################################

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
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
                texts = [" ".join([load_or_generate_kmer(data_path, [t], kmer)[0] for t in pair]) for pair in texts]
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
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }

@dataclass
class DataCollatorForSupervisedDataset:
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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

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
        return transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"Tokenizer not found in {tokenizer_path}.")
        return None

#########################################
#         MODEL & METRIC TOOLS          #
#########################################

def softmax(logits: np.ndarray):
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.detach().cpu()

def extended_compute_metrics(eval_pred):
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
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float32)

class CustomTrainer(transformers.Trainer):
    """Custom Trainer to use Focal Loss instead of CrossEntropyLoss."""
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
    """Logs training progress (loss, learning rate) at each logging step."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None and "loss" in logs:
            lr = logs.get('learning_rate', 'N/A')
            step_loss = logs['loss']
            logging.info(f"Step {state.global_step}: loss={step_loss:.4f}, lr={lr}")

#########################################
#        TRAINING & OPTUNA LOOP         #
#########################################

def build_and_train_model(model_args, training_args, tokenizer, train_dataset, val_dataset, data_collator, trial_params: dict = None):
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
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        except Exception as e:
            logger.warning(f"Failed to apply LoRA config: {e}")
            return None
    if model_args.use_class_weights:
        if trial_params:
            alpha = torch.tensor([trial_params["alpha_class_0"], trial_params["alpha_class_1"]])
            gamma = trial_params.get("gamma", 2.0)
        else:
            alpha = compute_class_weights(train_dataset.labels, train_dataset.num_labels)
            gamma = 2.0
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

def create_objective(model_args, training_args, tokenizer, train_dataset, val_dataset, data_collator, state_manager):
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
        combined_score = 0.4 * metrics.get("eval_auc", 0) + 0.6 * metrics.get("class_1_f1", 0)
        if trial.number == 0 or combined_score > trial.study.best_value:
            state_manager.save_best_model(trainer, trial_params)
        return combined_score
    return objective

#########################################
#                MAIN                 #
#########################################

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

    base_dir = training_args.output_dir
    tokenizer = get_latest_checkpoint_tokenizer(base_dir)
    if tokenizer is None:
        logger.info("Loading new tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        logger.info("Tokenizer created and loaded.")
    else:
        logger.info("Tokenizer successfully loaded from checkpoint.")

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    train_data_file = "train_oversampled.csv" if model_args.use_oversampled else "train.csv"
    train_data_path = os.path.join(data_args.data_path, train_data_file)
    logger.warning(f"Loading training dataset from: {train_data_path}")

    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_data_path,
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

    state_manager = ModelStateManager(run_name=f"DB2_{dataset_name}_Alt5", output_dir=str(results_dir))
    
    try:
        if model_args.use_optuna:
            study = optuna.create_study(direction="maximize")
            objective_fn = create_objective(model_args, training_args, tokenizer, train_dataset, val_dataset, data_collator, state_manager)
            study.optimize(objective_fn, n_trials=training_args.optuna_trials)
            
            model, best_params = state_manager.load_best_model(model_args, num_labels=train_dataset.num_labels)
            
            eval_trainer_class = CustomTrainer if model_args.use_class_weights else transformers.Trainer
            loss_fn = None
            if model_args.use_class_weights:
                alpha = torch.tensor([best_params["alpha_class_0"], best_params["alpha_class_1"]]).to(model.device)
                loss_fn = FocalLoss(alpha=alpha, gamma=best_params["gamma"])
            eval_trainer = eval_trainer_class(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                compute_metrics=extended_compute_metrics,
                loss_fn=loss_fn
            )
            final_results = eval_trainer.evaluate(test_dataset)
            with open(os.path.join(results_dir, "final_test_results.json"), "w") as f:
                json.dump(final_results, f)
            plot_confusion_matrix(eval_trainer, test_dataset, results_dir, f"DB2_{dataset_name}_Alt5")
            logger.info(f"Final Test Results: {final_results}")
        else:
            trainer = build_and_train_model(
                model_args=model_args,
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