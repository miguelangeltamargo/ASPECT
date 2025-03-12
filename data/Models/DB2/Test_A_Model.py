import os
import csv
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# LoRA imports
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="query,value")
    use_wandb: bool = field(default=False)
    use_class_weights: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    kmer: int = field(default=-1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./results")
    model_max_length: int = field(default=1024)
    per_device_eval_batch_size: int = field(default=32)
    fp16: bool = field(default=True)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)

def softmax(logits: np.ndarray):
    """Softmax for numpy array of shape [batch_size, num_classes]."""
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

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

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.detach().cpu()

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
            
        if len(data[0]) == 2:
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        else:
            raise ValueError("Data format not supported.")

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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
        )

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

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

def plot_confusion_matrix(trainer, eval_dataset, results_dir, runname_label):
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Constitutive", "Cassette"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {runname_label}")
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{runname_label}.png"))
    plt.close()

def get_latest_checkpoint_tokenizer(base_dir):
    """Check for the latest checkpoint tokenizer directory and load it."""
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        return None
    
    checkpoint_dir = checkpoint_dirs[0]
    tokenizer_path = os.path.join(base_dir, checkpoint_dir)
    
    if os.path.exists(tokenizer_path):
        return transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    return None

def load_and_evaluate():
    # Configuration
    dataset = "384_Split"
    base_dir = f"./results/DB2_{dataset}"
    model_dir = os.path.join(base_dir, "model_output")
    results_dir = Path(f"./results/DB2_{dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Model arguments
    model_args = ModelArguments()

    # Load tokenizer
    tokenizer = get_latest_checkpoint_tokenizer(model_dir)
    if tokenizer is None:
        logger.info("No saved tokenizer found, loading from original model...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=1024,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    # Load model
    try:
        checkpoint_dirs = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(model_dir, d))]
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoint directories found in {model_dir}")
        
        latest_checkpoint = checkpoint_dirs[0]
        model_path = os.path.join(model_dir, latest_checkpoint)
        
        logger.info(f"Loading model from {model_path}")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            trust_remote_code=True
        )

        # Apply LoRA configuration
        if model_args.use_lora:            
            try:
                # Try to use user-defined LoRA target modules
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
                model.print_trainable_parameters()
            except Exception as e:
                logging.warning(f"Failed to apply user-defined target_modules due to: {e}")
                logging.warning("Falling back to default target_modules=['Wqkv']")
                # Fall back to default LoRA config
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=["Wqkv"],  # Default module
                    lora_dropout=model_args.lora_dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    inference_mode=False,
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            logger.info("Applied LoRA configuration to model")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Load dataset
    data_args = DataArguments(
        data_path=f"./datasets/{dataset}/split_dataset/"
    )

    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(results_dir),
    )

    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        compute_metrics=extended_compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Evaluate
    logger.info("Starting evaluation...")
    try:
        results = trainer.evaluate(eval_dataset=test_dataset)
        
        # Save results
        results_file = results_dir / f"{dataset}_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_file}")

        # Plot confusion matrix
        plot_confusion_matrix(trainer, test_dataset, results_dir, f"Confusion_DB2_{dataset}_Evaluation")

        # Print metrics
        logger.info("\nKey Metrics:")
        logger.info(f"Accuracy: {results['eval_accuracy']:.4f}")
        logger.info(f"Binary F1: {results['eval_binary_f1']:.4f}")
        logger.info(f"Weighted F1: {results['eval_weighted_f1']:.4f}")
        logger.info(f"AUC: {results['eval_auc']:.4f}")
        logger.info(f"PR-AUC: {results['eval_pr_auc']:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    load_and_evaluate()