from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import pandas as pd


datasets = ['../datasets/Split2.0/SEQ_CONST.csv', 
            '../datasets/Split2.0/SEQ_ES.csv', 
            "../datasets/Split2.0/SEQ_3'.csv", 
            "../datasets/Split2.0/SEQ_5'.csv"]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])
print(combined_data.shape)
ogdata = pd.read_csv('../tNt/train_data.csv')
breakpoint()

# Split the data into features and target column
X = combined_data.drop('CLASS', axis=1)
y = combined_data['CLASS']

# Perform stratified sampling to maintain the class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

# Create a new DataFrame for the 20% subset with the same class distribution
subset_data = pd.concat([X_test, y_test], axis=1)

# Save the subset data to a CSV file
subset_data.to_csv('../tNt/subset_data.csv', index=False)