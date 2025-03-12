import numpy as np
import pandas as pd

file_path = "./datasets/512_Split/TeamsShare/Reduced_Inclusion_100/cons_8000_alt5/train.csv"

df_file = pd.read_csv(file_path)
print("Label 1 Count: " + str(len(df_file[df_file['label'] == 1])))
print("Label 0 Count: " + str(len(df_file[df_file['label'] == 0])))

file_path = "./datasets/512_Split/TeamsShare/Data/512_split_1_cons_alt5/train.csv"

df_file = pd.read_csv(file_path)
print("Label 1 Count: " + str(len(df_file[df_file['label'] == 1])))
print("Label 0 Count: " + str(len(df_file[df_file['label'] == 0])))