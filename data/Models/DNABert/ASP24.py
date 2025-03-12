#########################################
### Imports and Setup ###
#########################################

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, HF_dataset
from utils.viz_utils import count_plot, plot_confusion_matrix
from pathlib import Path
from optuna.pruners import MedianPruner
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import os
import optuna
import joblib
import warnings
from collections import Counter
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
#############################################
## Initialize variables and read data ##
#############################################

torch.manual_seed(42)
np.random.seed(42)

KMER = 5
NUM_FOLDS = 1
RANDOM_SEED = 42
SEQ_MAX_LEN = 512
EPOCHS = 10
BATCH_SIZE = 32
TRIALS = 3
TEST_SIZE = 0.20

# If you want a global way to skip oversampling for a particular run:
skip_oversampling = True  # Set True if you want to bypass ADASYN entirely for this test run

wandb.init(entity='mtamargo', project="ASPECT2024", name=f"DNABERT_{KMER}_Optuna")
wandb_config = {
    "model_path": f"ASPECT_{KMER}",
    "num_folds": NUM_FOLDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}
wandb.config.update(wandb_config)

results_dir = Path("./results") / "ASP" / f"ASP_RUN-Optuna"
results_dir.mkdir(parents=True, exist_ok=True)

f1_flog, acc_flog = {}, {}
sum_acc, sum_f1, eval_results = [], [], []

full_dataset = pd.read_csv("../../datasets/sahil/new_data_set/CSV/Resampled_100.csv")

train_set, test_set = train_test_split(
    full_dataset,
    test_size=TEST_SIZE,
    stratify=full_dataset["CLASS"],
    random_state=RANDOM_SEED
)

##########################################
### Prepare train+val portion ###
##########################################

ds_kmer, ds_labels = [], []
for seq, label in zip(train_set["SEQ"], test_set["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    ds_kmer.append(kmer_seq)
    ds_labels.append(label - 1)

df_kmers = np.array(ds_kmer)
df_labels = np.array(ds_labels)
NUM_CLASSES = len(np.unique(ds_labels))

model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}

tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])

############################
## Final Test Preparation ##
############################

test_kmer, test_labels = [], []
for seq, label in zip(test_set["SEQ"], test_set["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    test_kmer.append(kmer_seq)
    test_labels.append(label - 1)

test_kmer = np.array(test_kmer)
test_labels = np.array(test_labels)

test_kmer_list = test_kmer.tolist()
test_labels_list = test_labels.tolist()

test_encodings = tokenizer.batch_encode_plus(
    test_kmer_list,
    max_length=SEQ_MAX_LEN,
    truncation=True,
    padding=True,
    return_attention_mask=True,
    return_tensors="pt",
)

test_dataset = HF_dataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
    test_labels_list
)

################################
### Define the Objective Function for Optuna
################################

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    We'll do K-Fold on the train_val portion (80%).
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    assert learning_rate > 0, "Learning rate must be greater than zero."
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 5, 15)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64])
    gradient_checkpointing = trial.suggest_categorical('gradient_checkpointing', [True, False])

    trial_fold_acc = []
    trial_fold_f1 = []

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # We'll define these once outside the loop
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(KMER, KMER))
    adasyn = ADASYN(sampling_strategy='minority', random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_kmers, df_labels), 1):
        print(f"\n=== Trial {trial.number} - Fold {fold} ===")

        train_kmers_fold = df_kmers[train_idx]
        val_kmers_fold   = df_kmers[val_idx]
        train_labels_fold= df_labels[train_idx]
        val_labels_fold  = df_labels[val_idx]

        count_plot(train_labels_fold, f"Training Class Dist (Trial {trial.number} Fold {fold})", results_dir)

        train_kmers_list = train_kmers_fold.tolist()
        val_kmers_list   = val_kmers_fold.tolist()
        train_labels_list= train_labels_fold.tolist()
        val_labels_list  = val_labels_fold.tolist()

        # Vectorize the training k-mer strings -> numeric
        X_train_numeric = vectorizer.fit_transform(train_kmers_list).toarray()
        y_train_numeric = np.array(train_labels_list)

        # Check class distribution
        class_counts = Counter(y_train_numeric)
        print("Class distribution before ADASYN:", class_counts)

        # Decide whether to apply ADASYN
        # 1) If skip_oversampling is True, we skip
        # 2) Otherwise, we see if classes differ by at least some threshold
        #    e.g., 5% difference
        apply_adasyn = not skip_oversampling
        if apply_adasyn:
            # Check difference
            minority_class = min(class_counts.values())
            majority_class = max(class_counts.values())
            # e.g., 5% threshold
            threshold = 0.05 * (minority_class + majority_class) / 2
            diff = majority_class - minority_class

            if diff < threshold:
                print(f"Skipping ADASYN (fold {fold}) - Already near balanced (diff={diff}).")
                apply_adasyn = False
            else:
                print(f"Applying ADASYN (fold {fold}) - difference is {diff}.")

        if apply_adasyn:
            # Apply ADASYN
            X_resampled, y_resampled = adasyn.fit_resample(X_train_numeric, y_train_numeric)
            print("Shapes before ADASYN:", X_train_numeric.shape, y_train_numeric.shape)
            print("Shapes after  ADASYN:", X_resampled.shape, y_resampled.shape)

            # Convert back to text for your Hugging Face tokenizer
            X_resampled_text = vectorizer.inverse_transform(X_resampled)
            X_resampled_kmers = [" ".join(tokens) for tokens in X_resampled_text]
            y_resampled_list = y_resampled.tolist()
        else:
            # No oversampling
            X_resampled_kmers = train_kmers_list
            y_resampled_list  = train_labels_list

        # Now tokenize the training data
        train_encodings = tokenizer.batch_encode_plus(
            X_resampled_kmers,
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        val_encodings = tokenizer.batch_encode_plus(
            val_kmers_list,
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Build HF_dataset
        train_dataset_fold = HF_dataset(
            train_encodings["input_ids"],
            train_encodings["attention_mask"],
            y_resampled_list
        )
        val_dataset_fold = HF_dataset(
            val_encodings["input_ids"],
            val_encodings["attention_mask"],
            val_labels_list
        )

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_config["model_path"], 
            num_labels=NUM_CLASSES
        )

        training_args = TrainingArguments(
            output_dir=results_dir / f"trial_{trial.number}_fold_{fold}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=results_dir / f"trial_{trial.number}_fold_{fold}" / "logs",
            logging_steps=60,
            load_best_model_at_end=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            fp16=True,
            gradient_checkpointing=gradient_checkpointing,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset_fold,
            eval_dataset=val_dataset_fold,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train
        trainer.train()

        # Evaluate
        res = trainer.evaluate(val_dataset_fold)
        wandb.log({
            f"Trial {trial.number} Fold {fold} Acc": res["eval_accuracy"],
            f"Trial {trial.number} Fold {fold} F1": res["eval_f1"],
            f"Trial {trial.number} Fold {fold} Precision": res["eval_precision"],
            f"Trial {trial.number} Fold {fold} Recall": res["eval_recall"]
        })

        trial_fold_acc.append(res["eval_accuracy"])
        trial_fold_f1.append(res["eval_f1"])

        model_path = results_dir / f"trial_{trial.number}_fold_{fold}_model"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved at {model_path}")

        count_plot(val_labels_list, f"Eval Dist (Trial {trial.number} Fold {fold})", results_dir)
        plot_confusion_matrix(trainer, val_dataset_fold, fold, results_dir)

    avg_f1 = np.mean(trial_fold_f1)
    avg_acc = np.mean(trial_fold_acc)

    print(f"=== Trial {trial.number} Completed ===")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    wandb.log({
        f"Trial {trial.number} Average Acc": avg_acc,
        f"Trial {trial.number} Average F1": avg_f1
    })

    return avg_f1


####################################
### Optuna Hyperparam Optimization ###
####################################

study = optuna.create_study(
    direction="maximize",
    study_name="DNABERT_Hyperparameter_Optimization",
    pruner=MedianPruner()
)
study.optimize(objective, n_trials=TRIALS, timeout=None)
print("Best hyperparameters:", study.best_params)

joblib.dump(study, 'optuna_study.pkl')

print("\n--- FINAL TEST EVALUATION ON 20% DATA ---\n")
wandb.finish()