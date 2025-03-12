#########################################
### Importing the necessary libraries ###
#########################################

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, CustomTrainer, HF_dataset, dataset
from utils.viz_utils import count_plot
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
import pdb
import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# torch.cuda.set_max_split_size(1024)

KMER = 6
NUM_FOLDS = 5  # Number of folds for stratified k-fold cross-validation
RANDOM_SEED = 42  # Random seed for reproducibility
SEQ_MAX_LEN = 512  # max len of BERT
EPOCHS = 10
BATCH_SIZE = 16

#############################################
## Initializing variables and reading data ##
#############################################

# Initialize wandb
wandb.init(entity='mtamargo', project="ASPECT2024", name=f"DNABERT_{KMER}_F{NUM_FOLDS}")
wandb_config = {
	"model_path": f"ASPECT_{KMER}",
}
wandb.config.update(wandb_config)
# breakpoint()

results_dir = os.path.join(".", "results", "ASP")
os.makedirs(results_dir, exist_ok=True)
file_count = len(os.listdir(results_dir))
results_dir = Path(f"./results")/ "ASP" / f"ASP_RUN-{file_count}"
# results_dir = Path(f"./results"/f"ASPrun_{runm}|{file_count}")

f1_flog, acc_flog = {}, {}
sum_acc, sum_f1, eval_results = [], [], []                       # List to store evaluation results for each fold
tr_set = pd.read_csv("../../tNt/subset_data.csv")                # Load 20% subset of training data to split


# Split the data into 30% for testing and 70% for training, maintaining the same label distribution
train_set, test_set = train_test_split(tr_set, test_size=0.99, stratify=tr_set["CLASS"], random_state=42)

ds_kmer, ds_labels = [], []
for seq, label in zip(train_set["SEQ"], train_set["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)          # Convert sequence to k-mer representation
    ds_kmer.append(kmer_seq)                     # Append k-mer sequence to training data
    ds_labels.append(label - 1)                  # Adjust label indexing


# ds_kmer, ds_labels = [], []
# for seq, label in zip(tr_set["SEQ"], tr_set["CLASS"]):
#     kmer_seq = return_kmer(seq, K=KMER)                         # Convert sequence to k-mer representation
#     ds_kmer.append(kmer_seq)                                    # Append k-mer sequence to training data
#     ds_labels.append(label - 1)                                 # Adjust label indexing

df_kmers = np.array(ds_kmer)
df_labels = np.array(ds_labels)

# labels = data["CLASS"].values                                 # Isolate the label columns in dataframe
                                                                # not used as top block does this and more.
                                                                # need to implement a more original method
NUM_CLASSES = len(np.unique(ds_labels))
model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}
model, tokenizer, device = load_model(model_config, return_model=True)
# breakpoint()
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle = True)       # Setting up skf fold count
count = 0
# for train_idx, eval_idx in skf.split(                           # Splitting data into k-folds
#     ds_kmer, ds_labels):                                        # to isolate the train and test pairs
#         count+=1
#         train_kmers = [ds_kmer[idx] for idx in train_idx]
#         train_labels = [ds_labels[idx] for idx in train_idx]
#         eval_kmers = [ds_kmer[idx] for idx in eval_idx]
#         eval_labels = [ds_labels[idx] for idx in eval_idx]

for train_idx, eval_idx in skf.split(                            # Splitting data into k-folds
    df_kmers, df_labels):                                        # to isolate the train and test pairs
        count+=1
        # print("Train:",train_idx,'Test:',eval_idx)
        train_kmers, eval_kmers = [df_kmers[train_idx], df_kmers[eval_idx]]
        train_labels, eval_labels = [df_labels[train_idx], df_labels[eval_idx]]
        
        count_plot(train_labels, f"Training Class Distribution Fold {count}", results_dir)
        
        # breakpoint()
        # train_kmers = train_kmers.tolist()
        train_labels = train_labels.tolist()
        eval_kmers = eval_kmers.tolist()
        eval_labels = eval_labels.tolist()
                
        # Tokenize the two seperate data
        train_encodings = tokenizer.batch_encode_plus(
            train_kmers.tolist(),
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
  
        eval_encodings = tokenizer.batch_encode_plus(
            eval_kmers,
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
        # breakpoint()
        train_dataset = HF_dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)       #worked
        eval_dataset = HF_dataset(eval_encodings["input_ids"], eval_encodings["attention_mask"], eval_labels)           #worked

        # train_dataset = dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels, tokenizer)
        # test_dataset = dataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels, tokenizer)
        
        # Create DataLoader for the training dataset
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        # breakpoint()
        
        ############################################
        ### Training and evaluating the model #####
        ############################################
        
        
        results_dir.mkdir(parents=True, exist_ok=True)

        # Set up the Trainer
        training_args = TrainingArguments(
        output_dir=results_dir / f"fold_{count}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=results_dir / f"fold_{count}" / "logs",
        logging_steps=60,
        load_best_model_at_end=True,
	    evaluation_strategy="epoch",
	    save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True
        )
        
        trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=eval_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
        tokenizer=tokenizer,
        )
        
        # trainer = CustomTrainer(
        # model=model,                         # the instantiated 🤗 Transformers model to be trained
        # args=training_args,                  # training arguments, defined above
        # train_dataloader=train_dataloader,   # training dataset
        # eval_dataloader=eval_dataloader,     # evaluation dataset
        # compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
        # tokenizer=tokenizer,
        # )
        
        # Train and evaluate
        trainer.train()
        # breakpoint()
        # save the model and tokenizer
        model_path = results_dir / f"modelfold{count}"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # val_dataset_gene(tokenizer, kmer_size=KMER, maxl_len=512)       #dont need to call this here, cause kfolds
        count_plot(eval_labels, f"Evaluation Class Distribution Fold {count}", results_dir)
        #Evauating on test data of fold
        res = trainer.evaluate(eval_dataset)
        eval_results.append(res)
        
        # average over the eval_accuracy and eval_f1 from the dic items in eval_results
        fold_acc = np.mean([res["eval_accuracy"] for res in eval_results])
        fold_f1 = np.mean([res["eval_f1"] for res in eval_results])
        
        wandb.log({
        f"Fold {count} Acc" : fold_acc,
        f"Fold {count} F1" : fold_f1
        })
        acc_flog[f"Fold {count}"] = fold_acc
        f1_flog[f"Fold {count}"] = fold_f1
        
        # Update the sums and count
        # Append the averages to the respective lists
        sum_acc.append(fold_acc)
        sum_f1.append(fold_f1)

        print(f"Average accuracy fold-{count}: {fold_acc}")
        print(f"Average F1 fold-{count}: {fold_f1}")
       

# Calculate and print overall average after the loop ends
avg_acc = np.mean(sum_acc)
avg_f1 = np.mean(sum_f1)

print(f"Average accuracy: {avg_acc}")
print(f"Average F1: {avg_f1}")
print("\nsum_acc:", sum_acc)
print("\nsum_f1:", sum_f1)

# Log the lists
# wandb.log({"fold_acc": sum_acc, "fold_f1": sum_f1})
wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
wandb.finish()

print("\nDict Acc of Folds:", acc_flog)
print("\nDict F1 of Folds:", f1_flog)