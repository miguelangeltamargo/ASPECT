import pandas as pd

file_path = '../datasets/sahil/original_dataset_Constitutive_reducedto_4000.txt'
new_file = '../datasets/sahil/Original_Events_Reduced.csv'

data = pd.read_csv(file_path, sep='\t')

data.to_csv(new_file, index=False)

print(data.head())