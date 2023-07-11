import os
import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
from utils.data_utils import return_kmer, val_dataset_generator, HF_dataset
from utils.model_utils import load_model, compute_metrics
from utils.viz_utils import count_plot

# Define constants
KMER = 6
NUM_FOLDS = 5
RANDOM_SEED = 42
SEQ_MAX_LEN = 512

# Load the dataset
datasets = ['../QTest10.csv']
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=RANDOM_SEED, shuffle=True)

# Iterate over the folds
for fold, (train_index, val_index) in enumerate(skf.split(combined_data, combined_data['CLASS'])):
    # Split the data into train and validation sets
    train_data = combined_data.iloc[train_index]
    val_data = combined_data.iloc[val_index]

    # Save the train and validation data for each fold to separate files
    output_dir = f'./fold{fold+1}'
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # Prepare the train and validation datasets
    train_kmers, labels_train = [], []
    for seq, label in zip(train_data["SEQ"], train_data["CLASS"]):
        kmer_seq = return_kmer(seq, K=KMER)
        train_kmers.append(kmer_seq)
        labels_train.append(int(label) - 1)

    NUM_CLASSES = len(np.unique(labels_train))

    model_config = {
        "model_path": f"zhihan1996/DNA_bert_{KMER}",
        "num_classes": NUM_CLASSES,
    }

    # Load the model and tokenizer
    model, tokenizer, device = load_model(model_config, return_model=True)

    # Prepare the validation dataset
    val_kmers, labels_val = [], []
    for seq, label in zip(val_data["SEQ"], val_data["CLASS"]):
        kmer_seq = return_kmer(seq, K=KMER)
        val_kmers.append(kmer_seq)
        labels_val.append(label - 1)

    # Encode the training and validation datasets
    train_encodings = tokenizer.batch_encode_plus(
        train_kmers,
        max_length=SEQ_MAX_LEN,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    train_dataset = HF_dataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        labels_train,
    )

    val_encodings = tokenizer.batch_encode_plus(
        val_kmers,
        max_length=SEQ_MAX_LEN,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    val_dataset = HF_dataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        labels_val,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/checkpoints/fold{fold+1}",  # output directory
        num_train_epochs=20,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=60,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Evaluate the model
    eval_results = []
    eval_results_fold = trainer.evaluate(val_dataset)
    eval_results.append(eval_results_fold)
    print(f"Fold {fold+1} Evaluation Result: {eval_results_fold}")
    
    # Average the evaluation metrics over the test datasets for the fold
    avg_acc = np.mean([res["eval_accuracy"] for res in eval_results])
    avg_f1 = np.mean([res["eval_f1"] for res in eval_results])

    # Log the average accuracy and F1 score for the fold
    wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})

    # Save the fold's model and tokenizer
    model_path_fold = os.path.join(output_dir, f"fold{fold}", "model")
    model.save_pretrained(model_path_fold)
    tokenizer.save_pretrained(model_path_fold)

    # Append the fold's evaluation results to the overall evaluation results
    eval_results.append(eval_results_fold)

    # Increment the fold number
    fold += 1

# Calculate the average evaluation metrics over all folds
avg_acc = np.mean([np.mean([res["eval_accuracy"] for res in fold_results]) for fold_results in eval_results])
avg_f1 = np.mean([np.mean([res["eval_f1"] for res in fold_results]) for fold_results in eval_results])

print(f"Average accuracy across all folds: {avg_acc}")
print(f"Average F1 score across all folds: {avg_f1}")

wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
wandb.finish()