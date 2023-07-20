import pandas as pd
from pathlib import Path
from utils.data_utils import return_kmer, val_dataset_generator, HF_dataset
from utils.model_utils import load_model, compute_metrics
from utils.viz_utils import count_plot
import seaborn as sns
import matplotlib.pyplot as plt
KMER = 3  # The length of the K-mers to be used by the model and tokenizer

training_data_path = Path("../tNt/subset_data.csv")

df_training = pd.read_csv(training_data_path)
 
sns.barplot(x=df_training["CLASS"])

plt.title(f'Class Distribution In Training Set')
plt.savefig('tests.png')
plt.show()