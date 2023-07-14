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

data = pd.read_csv("../TrVaTe/train_data.csv")          # Load 80% of data to split

labels = data["CLASS"].values                           # Isolate the label columns in dataframe

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle = True)               # Setting up skf fold count
count = 0
for train_idx, test_idx in skf.split(                   # Splitting data into k-folds
    np.zeros(len(labels)), labels):                     # to isolate the train and test pairs
        # print(train_idx)
        # count+=1
        train_data = data.iloc[train_idx]
        test_data = data.loc[test_idx]

# print(count)
print(train_data)
# train_kmers, labels_train = [], []
# for seq, label in zip(df_training["SEQ"], df_training["CLASS"]):
#     kmer_seq = return_kmer(seq, K=KMER)
#     train_kmers.append(kmer_seq)
#     labels_train.append(label - 1)