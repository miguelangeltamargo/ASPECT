import pandas as pd
from sklearn.model_selection import train_test_split
path = "/aspect/ASPECT/data/datasets"
# List of dataset file names
datasets = ['/aspect/ASPECT/data/datasets/SEQ_CONST.csv', '/aspect/ASPECT/data/datasets/SEQ_ES.csv', "/aspect/ASPECT/data/datasets/SEQ_3'.csv", "/aspect/ASPECT/data/datasets/SEQ_5'.csv"]

# Combine all datasets into one DataFrame
combined_data = pd.concat([pd.read_csv(f) for f in datasets])

# Write the combined data to a single file
combined_data.to_csv('COMB_SET1.csv', index=False)

# Split the combined dataset into training+validation sets and testing sets
train_val_data, test_data = train_test_split(combined_data, test_size=0.2)

# # Further split the training+validation set into separate training and validation sets
train_data, val_data = train_test_split(train_val_data, test_size=0.25)

# Save the training, validation, and testing sets as separate CSV files
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)