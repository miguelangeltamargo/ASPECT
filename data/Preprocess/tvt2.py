import pandas as pd
from sklearn.model_selection import train_test_split

# List of dataset file names
datasets = ['../datasets/Split2.0/SEQ_CONST.csv', 
            '../datasets/Split2.0/SEQ_ES.csv', 
            "../datasets/Split2.0/SEQ_3'.csv", 
            "../datasets/Split2.0/SEQ_5'.csv"]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

# Split the combined dataset into training and testing sets
train_data, test_data = train_test_split(combined_data, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# Save the training and testing sets as separate CSV files
train_data.to_csv('../tNt/train_data.csv', index=False)
test_data.to_csv('../tNt/test/test_data.csv', index=False)
