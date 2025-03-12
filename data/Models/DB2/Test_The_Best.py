import os
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Union

import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Arguments for model evaluation"""
    model_path: str = field(default=None, metadata={"help": "Path to the saved model"})
    test_data_path: str = field(default=None, metadata={"help": "Path to test data CSV"})
    output_dir: str = field(default="./eval_results", metadata={"help": "Directory to save results"})
    batch_size: int = field(default=32, metadata={"help": "Batch size for evaluation"})
    kmer: int = field(default=-1, metadata={"help": "k-mer size, -1 means not using k-mer"})


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i: i + k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
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

class SupervisedDataset(torch.utils.data.Dataset):
    """Dataset for evaluation."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        
        # Load data from CSV (skipping header)
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
            
        if len(data[0]) == 2:
            try:
                texts = [d[0] for d in data]
                labels = [int(d[1]) for d in data]
            except ValueError:
                try:
                    texts = [d[1] for d in data]
                    labels = [int(d[0]) for d in data]
                except ValueError:
                    logger.error("Invalid data format")
                    raise
        else:
            raise ValueError("Data format not supported")

        if kmer != -1:
            logger.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

        # Tokenize the input texts
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


def softmax(logits: np.ndarray):
    """Compute softmax values for logits."""
    exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return logits.detach().cpu()  # Return tensor, not numpy


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def extended_compute_metrics(eval_pred):
    """Extended metrics computation including micro, macro, and weighted averages."""
    predictions, labels = eval_pred
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    if len(predictions.shape) == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
    elif len(predictions.shape) > 2:
        predictions = np.mean(predictions, axis=tuple(range(1, len(predictions.shape)-1)))
    
    try:
        pred_class = np.argmax(predictions, axis=1)
    except ValueError as e:
        logger.error(f"Error processing predictions shape {predictions.shape}: {e}")
        logger.error(f"Prediction sample: {predictions[:2]}")
        raise

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


def evaluate_model(args: EvalArguments):
    """Main function to evaluate a saved model."""
    logger.info("Starting model evaluation...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model, configuration, and tokenizer
    try:
        logger.info(f"Loading model and configuration from {args.model_path}")
        config = transformers.AutoConfig.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            num_labels=2
        )
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True
        )
        
        # Check for LoRA adapter configuration
        adapter_path = os.path.join(args.model_path, "adapter_config.json")
        if os.path.exists(adapter_path):
            logger.info("Found LoRA adapter configuration, loading PEFT model")
            from peft import PeftModel, PeftConfig  # Assumes these are installed and available
            peft_config = PeftConfig.from_pretrained(args.model_path)
            logger.info(f"LoRA config loaded: {peft_config}")
            model = PeftModel.from_pretrained(
                base_model,
                args.model_path,
                is_trainable=False
            )
            logger.info("Successfully loaded PEFT model with adapters")
        else:
            logger.warning("No LoRA adapter configuration found, using base model")
            model = base_model
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )
        
        training_args = transformers.TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False
        )
        
        # Move model to device and set to eval mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(f"Model path contents: {os.listdir(args.model_path)}")
        raise

    # Load dataset
    try:
        logger.info(f"Loading test data from {args.test_data_path}")
        test_dataset = SupervisedDataset(
            data_path=args.test_data_path,
            tokenizer=tokenizer,
            kmer=args.kmer
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Load model configuration to check for any saved parameters (e.g., focal loss)
    model_config = model.config.to_dict()
    logger.info("Model configuration loaded")
    alpha1 = model_config.get('alpha_class_0', None)
    alpha2 = model_config.get('alpha_class_1', None)
    gamma = model_config.get('gamma', None)

    # Instantiate trainer with a custom loss function if focal loss parameters are present.
    if alpha1 is not None and alpha2 is not None and gamma is not None:
        logger.info(f"Found saved focal loss parameters - Alpha1: {alpha1}, Alpha2: {alpha2}, Gamma: {gamma}")
        alpha = torch.tensor([alpha1, alpha2], dtype=torch.float32)
        alpha = alpha.to("cuda" if torch.cuda.is_available() else "cpu")
        # Assuming FocalLoss and CustomTrainer are defined elsewhere in your project.
        loss_fn = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            reduction='mean'
        )
        trainer = CustomTrainer(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=args.output_dir,
                per_device_eval_batch_size=args.batch_size,
                remove_unused_columns=False,
            ),
            tokenizer=tokenizer,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=extended_compute_metrics,
            loss_fn=loss_fn
        )
    else:
        logger.info("No focal loss parameters found, using default CrossEntropyLoss")
        trainer = transformers.Trainer(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=args.output_dir,
                per_device_eval_batch_size=args.batch_size,
                remove_unused_columns=False
            ),
            tokenizer=tokenizer,
            compute_metrics=extended_compute_metrics
        )

    # Run predictions
    logger.info("Running predictions...")
    output = trainer.predict(test_dataset)
    predictions = output.predictions
    true_labels = output.label_ids  # Use a different variable name to avoid conflicts
    metrics = output.metrics
    
    # Extract logits if predictions is a tuple or list
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    
    
    # probs_class1 = softmax(predictions)[:, 1]
    
    # best_thresh, best_f1 = 0.0, 0.0
    # for t in np.linspace(0,1,101):
    #     preds_t = (probs_class1 >= t).astype(int)
    #     f1_t = f1_score(labels, preds_t)
    #     if f1_t > best_f1:
    #         best_f1 = f1_t
    #         best_thresh = t
    
    # print(f"Optimal threshold={best_thresh:.2f}, F1={best_f1:.4f}")
    # with open(os.path.join(results_path, "best_threshold.json"), "w") as f:
    #     json.dump({"threshold": best_thresh, "f1": best_f1}, f)

    # probabilities = softmax(predictions)

    # Compute evaluation metrics
    logger.info("Computing metrics...")
    results = []
    results.append(metrics)
    # results["optimal_threshold"] = best_thresh
    # results["optimal_f1"] = best_f1

    # Save the evaluation results to a JSON file
    logger.info("Saving results...")
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_classes = np.argmax(predictions, axis=1)

    plot_confusion_matrix(
        true_labels,
        pred_classes,  # Use the predicted class labels, not the dataset object
        ["Constitutive", "Alternative 5"],
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # Instead of using a list, just work with the metrics dictionary
    logger.info("\nEvaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, (float, int)):
            logger.info(f"{metric}: {value:.4f}")
        else:
            logger.info(f"{metric}: {value}")

    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--test_data_path", required=True, help="Path to test data CSV")
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--kmer", type=int, default=-1, help="k-mer size (-1 for no k-mer)")
    
    args = parser.parse_args()
    eval_args = EvalArguments(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        kmer=args.kmer
    )
    
    evaluate_model(eval_args)
