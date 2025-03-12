
from transformers.models.bert.configuration_bert import BertConfig

config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

#########################################
### Imports and Setup ###
#########################################
import os
import sys
import json
import torch
import wandb
import optuna
import joblib
import logging
import warnings
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import ADASYN

from utils.model_utils import compute_metrics
from utils.data_utils import encode_data, get_adversarial_data

# Hugging Face
import transformers
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          TrainingArguments, 
                          Trainer, 
                          EarlyStoppingCallback)
from transformers.integrations import WandbCallback

# Optional: If you want the PyTorch lightning or accelerate 
# you'll need additional imports, but let's keep it to HF Trainer.

# If you have local utility functions, adapt the import below
# e.g., from utils.data_utils import get_adversarial_data, compute_metrics, encode_data
# For this example, we'll inline basic placeholders for them.

#########################################
### Local Utility Placeholders
#########################################

class DnaDataset(torch.utils.data.Dataset):
    """
    A simple Dataset class for the encoded examples.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items() if k != "labels"}
        item["labels"] = torch.tensor(self.encodings["labels"][idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings["labels"])

#########################################
### Main Training Script ###
#########################################
def main():
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, 
                        help="Path to your CSV dataset file with columns: SEQ, CLASS (int).")

    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M", 
                        help="Hugging Face model path for DNABERT2.")
    parser.add_argument("--cache_tokenized", action="store_true", 
                        help="Flag to cache tokenized data to disk.")
    parser.add_argument("--apply_adasyn", action="store_true", 
                        help="Flag to apply ADASYN oversampling on the minority class.")
    parser.add_argument("--adversarial_retrain", action="store_true", 
                        help="Flag to perform a second pass training on generated adversarial examples.")
    parser.add_argument("--n_trials", type=int, default=3, 
                        help="Number of Optuna trials.")
    parser.add_argument("--test_size", type=float, default=0.40, 
                        help="Test size ratio for train/test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        entity="mtamargo",
        project="ASPECT2024",
        name=f"DNABERT2_{Path(args.dataset_path).stem}_Optuna",
    )
    wandb.config.update(vars(args))

    # Fix seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Read dataset
    df = pd.read_csv(args.dataset_path)
    df["CLASS"] = df["CLASS"].astype(int)  # ensure integer labels

    # Basic split: 60/40 for train/(val+test), then 20/20 out of that 40 for val/test
    train_df, val_test_df = train_test_split(
        df, 
        test_size=args.test_size, 
        stratify=df["CLASS"],
        random_state=args.seed
    )
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=0.5,
        stratify=val_test_df["CLASS"],
        random_state=args.seed
    )

    # Prepare data arrays
    train_texts = train_df["SEQ"].values
    train_labels = train_df["CLASS"].values - 1  # if labels start from 1, shift to 0-based

    val_texts = val_df["SEQ"].values
    val_labels = val_df["CLASS"].values - 1

    test_texts = test_df["SEQ"].values
    test_labels = test_df["CLASS"].values - 1

    ##############################################################################
    ###   Tokenizer: from DNABERT2 (BPE-based). No k-mer needed.               ###
    ##############################################################################
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    # If you want a smaller max_length or a dynamic approach, define here:
    MAX_LENGTH = 512

    #####################################################################
    ###   Create a function to encode and optionally cache the data   ###
    #####################################################################
    def encode_and_cache(texts, labels, split_name="train"):
        cache_dir = Path("./token_cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{Path(args.dataset_path).stem}_{split_name}.pt"

        if args.cache_tokenized and cache_file.exists():
            logging.info(f"[CACHE] Loading cached {split_name} dataset from {cache_file}")
            return torch.load(cache_file)
        else:
            logging.info(f"[ENCODE] Tokenizing {split_name} data from scratch...")
            dataset = encode_data(texts, labels, tokenizer, max_length=MAX_LENGTH)
            encodings = {
                "input_ids": [item["input_ids"] for item in dataset],
                "attention_mask": [item["attention_mask"] for item in dataset],
                "labels": [item["labels"] for item in dataset],
            }
            if args.cache_tokenized:
                torch.save(encodings, cache_file)
                logging.info(f"[CACHE] Saved {split_name} dataset to {cache_file}")
            return encodings

    train_encodings = encode_and_cache(train_texts, train_labels, "train")
    val_encodings   = encode_and_cache(val_texts,   val_labels,   "val")
    test_encodings  = encode_and_cache(test_texts,  test_labels,  "test")

    #######################################################################
    ###   Build an Optuna objective for hyperparameter search           ###
    #######################################################################
    def objective(trial):
        # Suggest hyperparams
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        wd = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        num_epochs = trial.suggest_int('num_train_epochs', 3, 10)
        bs_train = trial.suggest_categorical('per_device_train_batch_size', [16, 32])
        grad_ckpt = trial.suggest_categorical('gradient_checkpointing', [True, False])

        # Possibly apply ADASYN
        if args.apply_adasyn:
            logging.info("[ADASYN] Oversampling minority class with ADASYN.")
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
            X_train_numeric = vectorizer.fit_transform(train_texts).toarray()
            y_train_numeric = train_labels

            class_counts = Counter(y_train_numeric)
            logging.info(f"Class distribution before ADASYN: {class_counts}")

            adasyn = ADASYN(random_state=args.seed)
            X_resampled, y_resampled = adasyn.fit_resample(X_train_numeric, y_train_numeric)

            # Convert back to strings
            X_res_text = vectorizer.inverse_transform(X_resampled)
            X_res_text = ["".join(tokens) for tokens in X_res_text]

            # Re-encode
            enc = encode_data(X_res_text, y_resampled, tokenizer, max_length=MAX_LENGTH)
            enc_train = {
                "input_ids": [e["input_ids"] for e in enc],
                "attention_mask": [e["attention_mask"] for e in enc],
                "labels": [e["labels"] for e in enc],
            }
        else:
            enc_train = train_encodings

        # Build datasets
        ds_train = DnaDataset(enc_train)
        ds_val   = DnaDataset(val_encodings)
        ds_test  = DnaDataset(test_encodings)

        # Load DNABERT2 model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=len(np.unique(train_labels)),  # adjust if your labels are 0-based
            trust_remote_code=True,
        )
        from transformers.models.bert.configuration_bert import BertConfig

        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

        # Set up Trainer
        output_dir = Path("./optuna_runs") / f"trial_{trial.number}"
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            learning_rate=lr,
            per_device_train_batch_size=bs_train,
            per_device_eval_batch_size=32,
            dataloader_pin_memory=True,
            dataloader_num_workers=10,
            warmup_steps=500,
            weight_decay=wd,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            gradient_checkpointing=grad_ckpt,
            fp16=True,
            report_to=["wandb"],  # Log to Weights & Biases
            run_name=f"DNABERT2_Trial_{trial.number}",
            seed=args.seed
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            tokenizer=tokenizer
        )

        # Train
        trainer.train()

        # Optional: Adversarial Retraining
        if args.adversarial_retrain:
            logging.info("[Adversarial] Generating adversarial samples from val set.")
            adv_texts, adv_labels = get_adversarial_data(model, tokenizer, ds_val)
            if len(adv_texts) > 0:
                logging.info(f"[Adversarial] Found {len(adv_texts)} adversarial examples. Retraining 1 epoch.")
                # Combine with original training data
                # If you used ADASYN, you have enc_train from above; otherwise train_encodings
                used_train_enc = enc_train if args.apply_adasyn else train_encodings

                # Convert used_train_enc back to arrays for simpler concatenation
                base_input_ids = used_train_enc["input_ids"]
                base_att_mask  = used_train_enc["attention_mask"]
                base_labels    = used_train_enc["labels"]

                # Encode adversarial data
                adv_enc = encode_data(adv_texts, adv_labels, tokenizer, max_length=MAX_LENGTH)
                adv_input_ids = [x["input_ids"] for x in adv_enc]
                adv_att_mask  = [x["attention_mask"] for x in adv_enc]
                adv_labels_   = [x["labels"] for x in adv_enc]

                # Concatenate
                final_input_ids = base_input_ids + adv_input_ids
                final_att_mask  = base_att_mask  + adv_att_mask
                final_labels    = base_labels    + adv_labels_

                ds_aug = DnaDataset({
                    "input_ids": final_input_ids,
                    "attention_mask": final_att_mask,
                    "labels": final_labels
                })

                trainer.train_dataset = ds_aug
                # Additional training epoch
                trainer.args.num_train_epochs = 1
                trainer.train()
            else:
                logging.info("[Adversarial] No adversarial samples found. Skipping extra training.")

        # Evaluate on test
        metrics = trainer.evaluate(ds_test)
        wandb.log({
            f"trial_{trial.number}_accuracy": metrics["eval_accuracy"],
            f"trial_{trial.number}_f1": metrics["eval_f1"],
            f"trial_{trial.number}_precision": metrics["eval_precision"],
            f"trial_{trial.number}_recall": metrics["eval_recall"]
            #Add AUC and AUC-PR
        })

        # Return metric that Optuna will maximize
        return metrics["eval_f1"]

    # Create study
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"DNABERT2_Study_{Path(args.dataset_path).stem}"
    )
    study.optimize(objective, n_trials=args.n_trials)

    logging.info(f"Best hyperparameters: {study.best_params}")

    # Optionally save the entire study
    joblib.dump(study, f"optuna_study_{Path(args.dataset_path).stem}.pkl")

    wandb.finish()
    logging.info("Finished all trials. End of script.")

if __name__ == "__main__":
    main()