#########################################
### Importing the necessary libraries ###
#########################################


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, val_dataset_gene, HF_dataset
from utils.viz_utils import count_plot
from pathlib import Path
import numpy as np
import pandas as pd
import wandb


KMER = 6
NUM_FOLDS = 5  # Number of folds for stratified k-fold cross-validation
RANDOM_SEED = 42  # Random seed for reproducibility
SEQ_MAX_LEN = 512  # max len of BERT



# Initialize wandb
wandb.init(project="DBertFolds", name=f"DNABERT_{KMER}_F{NUM_FOLDS}")

# Load your data
data = pd.read_csv("../TrVaTe/train_data.csv")

data["CLASS"] = data["CLASS"].astype(int)           # Convert labels to integers
if data["CLASS"].min() == 1:                        # Subtract 1 from labels if they start from 1
    data["CLASS"] -= 1

NUM_CLASSES = data["CLASS"].max() + 1               # Check the number of classes

labels = data["CLASS"].values                       

NUM_CLASSES = len(np.unique(labels))

count_plot(labels_train, "Training Class Distribution")

model_config = {
	"model_path": f"zhihan1996/DNA_bert_{KMER}",
	"num_classes": NUM_CLASSES,
}
wandb_config = {
	"model_path": f"DBertFolds_{KMER}",
	"num_classes": NUM_CLASSES,
}
wandb.config.update(wandb_config)

# Load the DNABERT model and tokenizer
model, tokenizer, device = load_model(model_config, return_model=True)

# Prepare data for StratifiedKFold
skf = StratifiedKFold(n_splits=NUM_FOLDS)

# List to store evaluation results for each fold
eval_results = []

for train_index, test_index in skf.split(np.zeros(len(labels)), labels):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    # Tokenize the data
    train_encodings = tokenizer(train_data["SEQ"].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_data["SEQ"].tolist(), truncation=True, padding=True)

    # Prepare the data for PyTorch
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(train_encodings, train_data["CLASS"].tolist())
    test_dataset = Dataset(test_encodings, test_data["CLASS"].tolist())


############################################
### Training and evaluating the model #####
############################################

    results_dir = Path("./resultsgpt/classification/")
    results_dir.mkdir(parents=True, exist_ok=True)
    EPOCHS = 15
    BATCH_SIZE = 10


    # Set up the Trainer
    training_args = TrainingArguments(
        output_dir=results_dir / "checkpoints",         # output directory
        num_train_epochs=EPOCHS,                        # total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,                 # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,                  # batch size for evaluation
        warmup_steps=500,                               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                              # strength of weight decay
        logging_dir=results_dir / "logsgpt",            # directory for storing logs
        logging_steps=60,
        load_best_model_at_end=True,
	    evaluation_strategy="epoch",
	    save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
        tokenizer=tokenizer,
    )

    # Train and evaluate
    trainer.train()

    # save the model and tokenizer
    model_path = results_dir / "model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Generate the validation dataset
    val_dataset = val_dataset_gene(tokenizer, KMER, test_data, SEQ_MAX_LEN)

    # Evaluate the model
    res = trainer.evaluate(val_dataset)
    eval_results.append(res)
    
    
    # Log metrics with wandb
    wandb.log({"FOLD_ACCURACY": res["EVAL_ACCURACY"], "Fold_F1": res["Eval_F1"]})

 
    
    
# Compute the average accuracy and F1 score
avg_acc = np.mean([res["EVAL_ACCURACY"] for res in eval_results])
avg_f1 = np.mean([res["Eval_F1"] for res in eval_results])

print(f"Average accuracy: {avg_acc}")
print(f"Average F1: {avg_f1}")

# Log average metrics with wandb
wandb.log({"AVG_ACCURACY": avg_acc, "AVG_F1": avg_f1})
wandb.finish()