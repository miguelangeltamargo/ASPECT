import pandas as pd

# Load a single CSV file
data = pd.read_csv('SEQ_CONST.csv')

# Check column names
print(data.columns)
