#########################################
### Imports and Setup ###
#########################################

import torch
# from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, encode_data, get_adversarial_data
from utils.viz_utils import count_plot, plot_confusion_matrix, plot_trial_confusion_matrix
from pathlib import Path
from optuna.pruners import MedianPruner
from imblearn.over_sampling import ADASYN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import wandb
import os
import optuna
import joblib
import warnings
from collections import Counter
from argparse import ArgumentParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


#############################################
## Initialize variables and read data ##
#############################################

torch.manual_seed(42)
np.random.seed(42)

KMER = 5
RANDOM_SEED = 42
SEQ_MAX_LEN = 512
EPOCHS = 10
BATCH_SIZE = 32
TRIALS = 3
EVAL_TEST_SIZE = 0.40

# If you want a global way to skip oversampling for a particular run:
skip_oversampling = True  # Set True if you want to bypass ADASYN entirely for this test run



#########################################
### parse arguments
#########################################
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file.")
args = parser.parse_args()



dataset_name = os.path.basename(args.dataset_path).replace(".csv","")

wandb.init(entity='mtamargo', project="ASPECT2024", name=f"DNABERT_{KMER}_QuickTest_{dataset_name}")
wandb_config = {
    "model_path": f"ASPECT_{KMER}_",
    "dataset": dataset_name,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
}
wandb.config.update(wandb_config)

results_dir = Path("./results") / "ASP" / f"QuickTest_{dataset_name}"
results_dir.mkdir(parents=True, exist_ok=True)

f1_flog, acc_flog = {}, {}
sum_acc, sum_f1, eval_results = [], [], []

full_dataset = pd.read_csv(args.dataset_path)
print(args.dataset_path)
###############################
#### Prepare Train Portion ####
###############################

train_data, val_test_data = train_test_split(       #Splits into 60%: Train & 40%: Test
    full_dataset,
    test_size=EVAL_TEST_SIZE,
    stratify=full_dataset["CLASS"],
    random_state=RANDOM_SEED
)

################################
### Prepare Val+Test Portion ###
################################

val_data, test_data = train_test_split(       #Splits into 20%: Validation & 20%: Test
    val_test_data,
    test_size=0.5,
    stratify=val_test_data["CLASS"],
    random_state=RANDOM_SEED
)

###########################
#### Train Preparation ####
###########################

train_kmers, train_labels = [], []
for seq, label in zip(train_data["SEQ"], train_data["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    train_kmers.append(kmer_seq)
    train_labels.append(label - 1)

train_kmers = np.array(train_kmers)
train_labels= np.array(train_labels)

############################
## Validation Preparation ##
############################

val_kmers, val_labels = [], []
for seq, label in zip(val_data["SEQ"], val_data["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    val_kmers.append(kmer_seq)
    val_labels.append(label - 1)

val_kmers = np.array(val_kmers)
val_labels= np.array(val_labels)

############################
## Final Test Preparation ##
############################

test_kmers, test_labels = [], []
for seq, label in zip(test_data["SEQ"], test_data["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    test_kmers.append(kmer_seq)
    test_labels.append(label - 1)

test_kmers = np.array(test_kmers)
test_labels= np.array(test_labels)



model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": len(np.unique(train_labels)),
}

tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"])

# train_dataset = encode_data(train_kmers, train_labels, tokenizer)
# val_dataset = encode_data(val_kmers, val_labels, tokenizer)
# test_dataset = encode_data(test_kmers, test_labels, tokenizer)

# =======================================
# Tokenize Once, Then Save/Load
# =======================================
tokenized_path = Path(f"./Token_Files/tokenized_{dataset_name}.pt")

if tokenized_path.exists():
    print(f"Loading tokenized datasets from {tokenized_path} ...")
    train_dataset, val_dataset, test_dataset = torch.load(tokenized_path)
else:
    print("Tokenizing datasets for the first time...")
    train_dataset = encode_data(train_kmers, train_labels, tokenizer)
    val_dataset = encode_data(val_kmers, val_labels, tokenizer)
    test_dataset = encode_data(test_kmers, test_labels, tokenizer)

    # Save to disk
    torch.save((train_dataset, val_dataset, test_dataset), tokenized_path)
    print(f"Tokenized datasets saved to {tokenized_path}.")


################################################
### Define the Objective Function for Optuna ###
################################################

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    assert learning_rate > 0, "Learning rate must be greater than zero."
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, EPOCHS)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [16, 32, 64])
    gradient_checkpointing = trial.suggest_categorical('gradient_checkpointing', [True, False])

    trial_acc = []
    trial_f1 = []
    
    ###############################################################
    ###### Oversampling with ADASYN With Threshhold Criteria ######
    ###############################################################
    
    # if not skip_oversampling:
    #     vectorizer = CountVectorizer(analyzer='char', ngram_range=(KMER, KMER))
    #     adasyn = ADASYN(sampling_strategy='minority', random_state=RANDOM_SEED)
        
    #     # **Vectorize Training Data**
    #     X_train_numeric = vectorizer.fit_transform(train_kmers.tolist()).toarray()
    #     y_train_numeric = train_labels
        
    #     class_counts = Counter(y_train_numeric)
    #     print("Class distribution before ADASYN:", class_counts)
        
    #     # **Decide Whether to Apply ADASYN**
    #     apply_adasyn = not skip_oversampling
    #     if apply_adasyn:
    #         minority_class = min(class_counts.values())
    #         majority_class = max(class_counts.values())
    #         threshold = 0.05 * ((minority_class + majority_class) / 2)
    #         diff = majority_class - minority_class
            
    #         if diff < threshold:
    #             print(f"Skipping ADASYN - Already near balanced (diff={diff}).")
    #             apply_adasyn = False
    #         else:
    #             print(f"Applying ADASYN - difference is {diff}.")
        
    #     if apply_adasyn:
    #         # **Apply ADASYN**
    #         X_resampled, y_resampled = adasyn.fit_resample(X_train_numeric, y_train_numeric)
    #         print("Shapes before ADASYN:", X_train_numeric.shape, y_train_numeric.shape)
    #         print("Shapes after  ADASYN:", X_resampled.shape, y_resampled.shape)
            
    #         # **Convert Back to Text for Tokenizer**
    #         X_resampled_text = vectorizer.inverse_transform(X_resampled)
    #         X_resampled_kmers = ["".join(tokens) for tokens in X_resampled_text]  # Corrected join
    #         y_resampled_labels = y_resampled.tolist()
            
    #         # **Encode Resampled Data**
    #         train_dataset = encode_data(X_resampled_kmers, y_resampled_labels, tokenizer)
    

    ##############################################################
    #### Oversampling with ADASYN without Threshhold Criteria ####
    ##############################################################
    
    if not skip_oversampling:
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(KMER, KMER))
        adasyn = ADASYN(sampling_strategy='minority', random_state=RANDOM_SEED)
        
        # **Vectorize Training Data**
        X_train_numeric = vectorizer.fit_transform(train_kmers.tolist()).toarray()
        y_train_numeric = train_labels
        
        class_counts = Counter(y_train_numeric)
        print("Class distribution before ADASYN:", class_counts)
        
        # **Apply ADASYN if needed**
        X_resampled, y_resampled = adasyn.fit_resample(X_train_numeric, y_train_numeric)
        print("Shapes before ADASYN:", X_train_numeric.shape, y_train_numeric.shape)
        print("Shapes after ADASYN:", X_resampled.shape, y_resampled.shape)
        
        # **Convert Back to Text for Tokenizer**
        X_resampled_text = vectorizer.inverse_transform(X_resampled)
        X_resampled_kmers = ["".join(tokens) for tokens in X_resampled_text]  # Corrected join
        y_resampled_labels = y_resampled.tolist()

        # **Encode Resampled Data**
        train_dataset = encode_data(X_resampled_kmers, y_resampled_labels, tokenizer)
    else:
        # **Skip ADASYN: Directly use the original dataset**
        train_dataset = encode_data(train_kmers, train_labels, tokenizer)



    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
    model_config["model_path"],
    num_labels=model_config["num_classes"]
)

    training_args = TrainingArguments(
        output_dir=results_dir / f"trial_{trial.number}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_pin_memory=True,
        dataloader_num_workers=10,
        warmup_steps=500,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=results_dir / f"trial_{trial.number}" / "logs",
        logging_steps=60,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=gradient_checkpointing,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    trainer.train()
    
    
    # Gather adversarial examples from the validation set
    adversarial_kmers, adversarial_labels = get_adversarial_data(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=val_dataset
    )
    print(f"Found {len(adversarial_kmers)} adversarial examples.")
    
    # Append adversarial examples to the training data if any
    if len(adversarial_kmers) > 0:
        # Combine with original training k-mers and labels
        augmented_kmers = np.concatenate([train_kmers, adversarial_kmers], axis=0)
        augmented_labels = np.concatenate([train_labels, adversarial_labels], axis=0)

        # Re-encode the augmented dataset
        train_dataset_aug = encode_data(augmented_kmers, augmented_labels, tokenizer)

        # Update the Trainer's training dataset
        trainer.train_dataset = train_dataset_aug

        # Optionally, perform additional training epochs with augmented data
        # Here, we train for 1 more epoch
        trainer.args.num_train_epochs = 1
        trainer.train()
    
    # Evaluate on the test set
    res = trainer.evaluate(test_dataset)
    wandb.log({
        f"Trial {trial.number} Acc": res["eval_accuracy"],
        f"Trial {trial.number} F1": res["eval_f1"],
        f"Trial {trial.number} Precision": res["eval_precision"],
        f"Trial {trial.number} Recall": res["eval_recall"]
    })
    
    trial_acc.append(res["eval_accuracy"])
    trial_f1.append(res["eval_f1"])
    
    # Save the model
    model_path = results_dir / f"trial_{trial.number}_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved at {model_path}")
    
    # Visualization
    count_plot(test_labels, f"Eval Dist (Trial {trial.number})", results_dir)
    plot_trial_confusion_matrix(trainer, test_dataset, trial.number, results_dir)
    
    # Compute average metrics
    avg_f1 = np.mean(trial_f1)
    avg_acc = np.mean(trial_acc)
    
    print(f"=== Trial {trial.number} Completed ===")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    wandb.log({
        f"Trial {trial.number} Average Acc": avg_acc,
        f"Trial {trial.number} Average F1": avg_f1
    })
    
    return res["eval_accuracy"]


######################################
### Optuna Hyperparam Optimization ###
######################################

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