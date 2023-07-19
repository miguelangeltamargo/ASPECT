#########################################
### Importing the necessary libraries ###
#########################################


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from utils.model_utils import load_model, compute_metrics
from utils.data_utils import return_kmer, val_dataset_gene, HF_dataset, dataset
from utils.viz_utils import count_plot
from pathlib import Path
import numpy as np
import pandas as pd
import wandb
import pdb
import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"



KMER = 6
NUM_FOLDS = 5  # Number of folds for stratified k-fold cross-validation
RANDOM_SEED = 42  # Random seed for reproducibility
SEQ_MAX_LEN = 512  # max len of BERT

#############################################
## Initializing variables and reading data ##
#############################################

# Initialize wandb
wandb.init(project="DBertFolds", name=f"DNABERT_{KMER}_F{NUM_FOLDS}")
wandb_config = {
	"model_path": f"DBertFolds_{KMER}",
}
# wandb.config.update(wandb_config)

# Access the run number
runm = wandb.run.id
sum_acc, sum_f1, eval_results = [], [], []
eval_results = []                                               # List to store evaluation results for each fold
tr_set = pd.read_csv("../TrVaTe/train_data.csv")                # Load 80% of data to split
ds_kmer, ds_labels = [], []
for seq, label in zip(tr_set["SEQ"], tr_set["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)                         # Convert sequence to k-mer representation
    ds_kmer.append(kmer_seq)                                    # Append k-mer sequence to training data
    ds_labels.append(label - 1)                                 # Adjust label indexing


# labels = data["CLASS"].values                                 # Isolate the label columns in dataframe
                                                                # not used as top block does this and more.
                                                                # # need to implement a more original method
NUM_CLASSES = len(np.unique(ds_labels))
model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}
model, tokenizer, device = load_model(model_config, return_model=True)
breakpoint()
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle = True)       # Setting up skf fold count
count = 0
for train_idx, test_idx in skf.split(                           # Splitting data into k-folds
    ds_kmer, ds_labels):                       # to isolate the train and test pairs

# for train_idx, test_idx in skf.split(                           # Splitting data into k-folds
    # np.zeros(len(ds_labels)), ds_labels):                       # to isolate the train and test pairs
        # print(train_idx)
        count+=1
        # trf_data = tr_set.iloc[train_idx]                       # training fold data of index labels and corresponding seq?
        # tef_data = tr_set.loc[test_idx]                         # testing fold data of index labels and corresponding seq
        
        train_kmers = [ds_kmer[idx] for idx in train_idx]
        train_labels = [ds_labels[idx] for idx in train_idx]
        test_kmers = [ds_kmer[idx] for idx in test_idx]
        test_labels = [ds_labels[idx] for idx in test_idx]
        count_plot(train_labels, f"Training Class Distribution Fold {count}")
        # Tokenize the two seperate data
        train_encodings = tokenizer.batch_encode_plus(
            train_kmers,
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
  
        test_encodings = tokenizer.batch_encode_plus(
            test_kmers,
            max_length=SEQ_MAX_LEN,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",  # return pytorch tensors
        )
        # breakpoint()
        train_dataset = HF_dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels)       #worked
        test_dataset = HF_dataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels)           #worked

        # train_dataset = dataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels, tokenizer)
        # test_dataset = dataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_labels, tokenizer)

        # breakpoint()
        
        ############################################
        ### Training and evaluating the model #####
        ############################################
        
        
        results_dir = Path("./results/classification/")     #change directory
        results_dir.mkdir(parents=True, exist_ok=True)
        EPOCHS = 10
        BATCH_SIZE = 16

        # Set up the Trainer
        training_args = TrainingArguments(
        output_dir=results_dir / f"testrun_{runm}"/ f"fold_{count}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=results_dir / f"testrun-{runm}"/ f"fold_{count}" / "logs",
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
        eval_dataset=test_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,     # computing metrics for evaluation in wandb
        tokenizer=tokenizer,
        )
        
        # Train and evaluate
        trainer.train()
        breakpoint()
        # save the model and tokenizer
        model_path = results_dir /f"testrun-{runm}"/f"modelfold{count}"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # val_dataset_gene(tokenizer, kmer_size=KMER, maxl_len=512)       #dont need to call this here, cause kfolds
        count_plot(test_labels, f"Testing Class Distribution Fold {count}")
        #Evauating on test data of fold
        res = trainer.evaluate(test_dataset)
        eval_results.append(res)
        
        # average over the eval_accuracy and eval_f1 from the dic items in eval_results
        fold_acc = np.mean([res["eval_accuracy"] for res in eval_results])
        fold_f1 = np.mean([res["eval_f1"] for res in eval_results])
    
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
 
wandb.log({"avg_acc": avg_acc, "avg_f1": avg_f1})
wandb.finish()