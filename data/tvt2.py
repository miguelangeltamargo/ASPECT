import pandas as pd
from sklearn.model_selection import train_test_split

# List of dataset file names
datasets = ['/aspect/ASPECT/data/datasets/SEQ_CONST.csv', '/aspect/ASPECT/data/datasets/SEQ_ES.csv', "/aspect/ASPECT/data/datasets/SEQ_3'.csv", "/aspect/ASPECT/data/datasets/SEQ_5'.csv"]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

# Split the combined dataset into training and testing sets
train_data, test_data = train_test_split(combined_data, test_size=0.2)

# Save the training and testing sets as separate CSV files
train_data.to_csv('./TrVaTe/train_data.csv', index=False)
test_data.to_csv('./TrVaTe/test/test_data.csv', index=False)
