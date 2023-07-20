import pandas as pd
from sklearn.model_selection import train_test_split

# List of dataset file names
<<<<<<< HEAD:data/preprocess/tvt2.py
datasets = ['../datasets/Split2.0/SEQ_CONST.csv', 
            '../datasets/Split2.0/SEQ_ES.csv', 
            "../datasets/Split2.0/SEQ_3'.csv", 
            "../datasets/Split2.0/SEQ_5'.csv"]
=======
datasets = ['/aspect/ASPECT/data/datasets/SEQ_CONST.csv', '/aspect/ASPECT/data/datasets/SEQ_ES.csv', "/aspect/ASPECT/data/datasets/SEQ_3'.csv", "/aspect/ASPECT/data/datasets/SEQ_5'.csv"]
>>>>>>> 979cf5dc1e66d97956b2cf2ce05f1f8d7cc1c116:data/tvt2.py

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

# Split the combined dataset into training and testing sets
train_data, test_data = train_test_split(combined_data, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


# Save the training and testing sets as separate CSV files
<<<<<<< HEAD:data/preprocess/tvt2.py
train_data.to_csv('../tNt/train_data.csv', index=False)
test_data.to_csv('../tNt/test/test_data.csv', index=False)
=======
train_data.to_csv('./TrVaTe/train_data.csv', index=False)
test_data.to_csv('./TrVaTe/test/test_data.csv', index=False)
>>>>>>> 979cf5dc1e66d97956b2cf2ce05f1f8d7cc1c116:data/tvt2.py
