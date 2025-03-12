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

# ADASYN
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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M") 
    # model_name_or_path: Optional[str] = field(default="facebook/opt-125m") 
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})

    # New flags
    use_wandb: bool = field(default=False, metadata={"help": "Whether to log to Weights & Biases"})
    use_optuna: bool = field(default=False, metadata={"help": "Whether to run hyperparameter search with Optuna"})
    use_class_weights: bool = field(default=False, metadata={"help": "Whether to apply class weights"})
    apply_adasyn: bool = field(default=False, metadata={"help": "Whether to apply ADASYN oversampling"})


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
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    warmup_steps: int = field(default=500)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    load_best_model_at_end: bool = field(default=True)
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=10)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_weighted_f1")
    greater_is_better: bool = field(default=True)
    save_total_limit: int = field(default=1)
    seed: int = field(default=42)
    # For Optuna
    optuna_trials: int = field(default=1, metadata={"help": "Number of trials for Optuna HPO"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # Save the model configuration
        trainer.model.config.save_pretrained(output_dir)
        
        # Save the model weights
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        
        # If using LoRA, save the LoRA configuration
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
        # apply_adasyn: bool = False,
    ):
        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")

        # # Optionally apply ADASYN oversampling (binary or multi-class)
        # if apply_adasyn:
        #     logging.warning("Applying ADASYN oversampling on the minority class...")
        #     # Flatten 'texts' if it is list of lists:
        #     if isinstance(texts[0], list):
        #         # If 2-sequence input, join them for oversampling. 
        #         # Or handle with caution for multi-seq scenarios
        #         texts_flat = [" ".join(t) for t in texts]
        #     else:
        #         texts_flat = texts

        #     # Convert text -> numeric features for ADASYN
        #     from sklearn.feature_extraction.text import CountVectorizer
        #     vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
        #     X = vectorizer.fit_transform(texts_flat).toarray()
        #     y = np.array(labels)

        #     class_counts = Counter(y)
        #     logging.warning(f"Class distribution before ADASYN: {class_counts}")

        #     adasyn = ADASYN(random_state=42)
        #     X_res, y_res = adasyn.fit_resample(X, y)
        #     # Convert back to strings
        #     X_res_text = vectorizer.inverse_transform(X_res)
        #     X_res_text = ["".join(tokens) for tokens in X_res_text]

        #     # Reassign
        #     texts = X_res_text
        #     labels = y_res.tolist()

        if kmer != -1:
            # only write file on the first process
            if torch.distributed.is_initialized() and torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            # If texts is list of lists, we do k-mer for each piece, 
            # but often DNABERT expects single seq -> adapt as needed
            if isinstance(texts[0], list):
                # For pair classification, generate k-mer separately
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
    """
    Check for the latest checkpoint tokenizer directory and load it if available.
    
    Args:
        base_dir (str): Base directory where checkpoints are stored (e.g., '/dataset/512/model_output/').
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer): Loaded tokenizer if available, else None.
    """
    
    if not Path(base_dir).is_dir():
        print(f"Base directory does not exist: {base_dir}")
        return None
    
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {base_dir}.")
        return None
    
    # Get the only checkpoint directory (assuming there's only one)
    checkpoint_dir = checkpoint_dirs[0]
    tokenizer_path = os.path.join(base_dir, checkpoint_dir, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    else:
        print(f"Tokenizer not found in {tokenizer_path}.")
        return None


def extended_compute_metrics(eval_pred):
    """
    Extended compute_metrics with ROC, PR, calibration, plus standard F1, precision, recall, etc.
    """
    predictions, labels = eval_pred

    # Convert logits to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    # Argmax classification
    pred_class = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, pred_class)
    
    # Binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="binary", pos_label=1, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, pred_class, average="weighted", zero_division=0
    )
    per_class_metrics = precision_recall_fscore_support(labels, pred_class, average=None, zero_division=0)

    # Check if binary
    num_classes = len(np.unique(labels))
    auc_value, pr_auc_value = float("nan"), float("nan")
    # We'll store optional curves in the returned dict
    roc_dict, pr_dict, calibration_dict = None, None, None

    if num_classes == 2:
        # Probability for class=1
        probs_class1 = softmax(predictions)[:, 1]
        auc_value = roc_auc_score(labels, probs_class1)
        pr_auc_value = average_precision_score(labels, probs_class1)

        # **Add ROC curve** (fpr, tpr)
        fpr, tpr, _ = roc_curve(labels, probs_class1)
        roc_dict = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        # **Add PR curve** (precision, recall)
        prec_arr, rec_arr, _ = precision_recall_curve(labels, probs_class1)
        pr_dict = {"precision": prec_arr.tolist(), "recall": rec_arr.tolist()}

        # **Add calibration curve** (prob_true, prob_pred) with 10 bins
        prob_true, prob_pred = calibration_curve(labels, probs_class1, n_bins=10)
        calibration_dict = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }

    class_report = classification_report(
        labels, pred_class, target_names=["Constitutive", "Alternative 5"], output_dict=True
    )

    return {
        "accuracy": acc,

        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": f1,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,

        "per_class_precision": per_class_metrics[0].tolist(),
        "per_class_recall": per_class_metrics[1].tolist(),
        "per_class_f1": per_class_metrics[2].tolist(),

        "auc": auc_value,
        "pr_auc": pr_auc_value,

        # "roc_curve": roc_dict,
        # "pr_curve": pr_dict,
        # "calibration_curve": calibration_dict,

        "class_report": class_report
    }


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

# Define function to plot confusion matrix
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
    

# Define the FocalLoss Class
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        
        Args:
            alpha (torch.Tensor, optional): Class weights. Shape [num_classes].
            gamma (float, optional): Focusing parameter.
            reduction (str, optional): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.alpha, reduction='none')  # We'll handle reduction

    def forward(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)  # Shape [batch_size]
        pt = torch.exp(-ce_loss)  # Probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Compute the inverse frequency of each class
def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights as the inverse frequency of each class.
    
    Args:
        labels (List[int]): List of class labels.
        num_classes (int): Number of classes.
        
    Returns:
        torch.Tensor: Class weights tensor.
    """
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    weights = torch.tensor(class_weights, dtype=torch.float32)
    return weights

# Create a Custom Trainer
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
     # Setup logging (enhanced)
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
    # results_dir = Path("./results") / f"DB2_{dataset_name}_OverSampled" if model_args.apply_adasyn else f"DB2_{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Update output directory and logging directory
    training_args.output_dir = str(results_dir / "model_output")
    training_args.logging_dir = str(results_dir / "logs")
    
    if model_args.use_wandb:
        # Initialize Weights & Biases
        run_name = f"DB2_{dataset_name}_Alt5"
        wandb.init(entity='mtamargo', project="ASPECT2", name=f"{run_name}")
        wandb.config.update({**vars(model_args), **vars(data_args), **vars(training_args)})
    else:
        run_name = "no_wandb_run"  # Default run name when WandB is not used


    # Set random seed
    torch.manual_seed(training_args.seed)
    np.random.seed(training_args.seed)


    base_dir = training_args.output_dir                     # using trianing args output directory to find tokenizer json
    tokenizer = get_latest_checkpoint_tokenizer(base_dir)

    if tokenizer is None:
        # Recompute and save the tokenizer here
        print("Computing new tokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
        print("Loaded Tokenizer from json successfully :D")
        # tokenizer.save_pretrained(os.path.join(base_dir, "checkpoint_new/tokenizer"))
        # print(f"New tokenizer saved to {os.path.join(base_dir, 'checkpoint_new/tokenizer')}")
    else:
        print("Tokenizer successfully loaded.")


    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    
    # Build CSV file to load based on the apply_adasyn flag
    train_data_path = os.path.join(
            data_args.data_path, 
            "train_oversampled.csv" if model_args.apply_adasyn else "train.csv"
        )

    logging.warning(f"Loading dataset: {train_data_path}")

    # Load dataset
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=train_data_path,
        kmer=data_args.kmer,
        # apply_adasyn=False,  # Make sure we do NOT run ADASYN inside this constructor
    )

    val_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "dev.csv"),
        kmer=data_args.kmer,
        # apply_adasyn=False,  # Only oversample the training set
    )
    test_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=os.path.join(data_args.data_path, "test.csv"),
        kmer=data_args.kmer,
        # apply_adasyn=False,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # 4. Define the Training Function with Focal Loss
    # ============================
    def build_and_train_model(learning_rate, weight_decay, num_train_epochs, batch_size, alpha=None, gamma=2.0):
        """Helper function for normal or Optuna-based training loops."""
        # load model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
        )

        # configure LoRA
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


        if model_args.use_class_weights:
            # Compute class weights if alpha is not provided
            if alpha is None:
                class_weights = compute_class_weights(train_dataset.labels, train_dataset.num_labels)
            else:
                class_weights = alpha

            class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize Focal Loss
            loss_fn = FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
        
        
        # Override training_args with possible optuna suggestions
        local_training_args = copy.deepcopy(training_args)
        local_training_args.learning_rate = learning_rate
        local_training_args.weight_decay = weight_decay
        local_training_args.num_train_epochs = num_train_epochs
        local_training_args.per_device_train_batch_size = batch_size

        # Define trainer with focal loss or with cross-entropy loss
        if model_args.use_class_weights:
            # Define custom trainer with Focal Loss
            trainer = CustomTrainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                loss_fn=loss_fn,  # Pass the focal loss function
                # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
            )
        else:
            # Define custom trainer with standard cross-entropy loss
            trainer = transformers.Trainer(
                model=model,
                tokenizer=tokenizer,
                args=local_training_args,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                compute_metrics=extended_compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                # callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
        )
            
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

        if local_training_args.save_model:
            # Save the full model state including configuration
            trainer.save_state()
            trainer.save_model()  # This will save the full model including config
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=local_training_args.output_dir)
            
        # Evaluate on test
        if local_training_args.eval_and_save_results:
            results_path = os.path.dirname(local_training_args.output_dir)            
            # results_path = os.path.join(local_training_args.output_dir, "results")
            results = trainer.evaluate(eval_dataset=test_dataset)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, f"{dataset_name}_eval_results.json"), "w") as f:
                json.dump(results, f)
                
            plot_confusion_matrix(trainer, test_dataset, results_path, run_name)
            
        return trainer



    try:
        if model_args.use_optuna:
            # ===============================
            #   Hyperparameter Search
            # ===============================
            def objective(trial):
                trial_number = trial.number
                logger.info(f"Starting trial number: {trial_number}")
                if model_args.use_wandb:
                    wandb.log({"trial_number": trial_number})
                # Example hyperparameter search space
                lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                epochs = trial.suggest_int("num_train_epochs", 7, 10)
                bs = trial.suggest_categorical("batch_size", [16, 32])

                # Sample gamma for Focal Loss
                gamma = trial.suggest_float("gamma", 3, 5.0)
                # Sample alpha for Focal Loss
                alpha1 = trial.suggest_float("alpha_class_0", 0.1, 1.0)  # constitutive class
                alpha2 = trial.suggest_float("alpha_class_1", 0.1, 1.0)  # minority class
                alpha = torch.tensor([alpha1, alpha2], dtype=torch.float32)

                # Train
                trainer = build_and_train_model(lr, wd, epochs, bs, alpha=alpha, gamma=gamma)

                # Evaluate
                metrics = trainer.evaluate(test_dataset)
                logger.info(f"Trial {trial_number} metrics: {metrics}")
                # We'll maximize F1 or AUC
                return metrics["eval_weighted_f1"]  # or metrics["eval_pr_auc"] if preferred

            # Runs Hyper-Parameter Search Model Loop
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=training_args.optuna_trials)

            logger.info(f"Optuna finished. Best params: {study.best_params}")
            
            # Logging best Optuna Params 
            try:
                output_dir = f"./results/{run_name}/"
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"{run_name}_best_params.json"), "w") as f:
                    json.dump(study.best_params, f)
                logger.info(f"Best params saved at {output_dir}/{run_name}_best_params.json")
            except Exception as e:
                logger.error(f"An error occurred writing best params to disk: {e}")
            
            logger.info(f"Best params to write: {study.best_params}")
            if model_args.use_wandb:
                wandb.log({"best_optuna_params": study.best_params})

        else:
            # ===========================
            #   Normal Single Training
            # ===========================
            build_and_train_model(
                learning_rate=training_args.learning_rate,
                weight_decay=training_args.weight_decay,
                num_train_epochs=training_args.num_train_epochs,
                batch_size=training_args.per_device_train_batch_size,
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted manually (Ctrl+C). Logging and stopping WandB run...")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

    finally:
        if model_args.use_wandb:
            wandb.finish()  # Make sure WandB run is properly closed


if __name__ == "__main__":
    train()
