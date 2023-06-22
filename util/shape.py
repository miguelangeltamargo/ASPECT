import numpy as np

data = np.genfromtxt('filtered_set.txt', delimiter='\t', dtype=object, skip_header=1, usecols=range(14), encoding='cp1252')

print('Data shape:', data.shape)