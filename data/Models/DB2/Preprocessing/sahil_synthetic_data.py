# creating synthetic data
# cassette
from collections import Counter
from imblearn.over_sampling import ADASYN
from matplotlib import pyplot as plt
from numpy import where
import pandas as pd
import random

OFFSET = 512
SAMPLE_SIZE = 0.8

# File paths
input_file = f"../datasets/{OFFSET}_Split/split_dataset_5_{SAMPLE_SIZE*100:g}_Balanced/train.csv"
output_file = f"../datasets/{OFFSET}_Split/split_dataset_5_{SAMPLE_SIZE*100:g}_Balanced/train_{SAMPLE_SIZE*100:g}_Balanced.csv"

# Load the dataset
df = pd.read_csv(input_file)

# Ensure the label column is numeric
df['label'] = df['label'].astype(int)

# Feature engineering: calculate GT content and AG content
def gt_content(sequence):
    gt_count = sequence.count('G') + sequence.count('T')
    return gt_count / len(sequence) if len(sequence) > 0 else 0

def ag_content(sequence):
    ag_count = sequence.count('A') + sequence.count('G')
    return ag_count / len(sequence) if len(sequence) > 0 else 0

df['GT_Content'] = df['sequence'].apply(gt_content)
df['AG_Content'] = df['sequence'].apply(ag_content)

# Create an Event_Class_Label column (here, the same as 'label')
df['Event_Class_Label'] = df['label']

# Extract features (X) and labels (y)
X = df[['GT_Content', 'AG_Content']].values
y = df['Event_Class_Label'].values

# Summarize class distribution
counter = Counter(y)
print("Original Class Distribution:", counter)

# Define sampling strategy:
# For SAMPLE_SIZE = 1, oversample minority (label 1) to equal the count of majority (label 0)
sampling_strategy = {0: counter[0], 1: int(SAMPLE_SIZE * counter[0])}

# Apply ADASYN oversampling
oversample = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Summarize the new class distribution
counter_res = Counter(y_resampled)
print("New Class Distribution:", counter_res)

# Create a DataFrame for the resampled data
resampled_df = pd.DataFrame(X_resampled, columns=['GT_Content', 'AG_Content'])
resampled_df['Event_Class_Label'] = y_resampled

# Map back to the original format:
# For indices within the original data, use the original sequence;
# for synthetic rows (index >= len(df)), use a random original sequence as a placeholder.
synthetic_data = []
original_sequences = df['sequence'].tolist()

for index, row in resampled_df.iterrows():
    # Use the oversampled label directly
    label = row['Event_Class_Label']
    # For synthetic rows, you might generate or select a sequence that
    # corresponds to the oversampled feature; here we simply pick a random one.
    # But note: this means your synthetic sequence doesn't truly reflect the oversampled features.
    seq = (df.iloc[index]['sequence']
           if index < len(df)
           else random.choice(original_sequences))
    synthetic_data.append({
        'label': label,  # use the numeric label from oversampling
        'sequence': seq
    })

final_df = pd.DataFrame(synthetic_data)

print("Label 1 Count: " + str(len(final_df[final_df['label'] == 1])))

print(f"Dataset split complete!")
print(f"Training set: {len(final_df)} samples -> {output_file}")

# Print data distribution
print("\nClass distribution in training set:")
print(final_df['label'].value_counts(normalize=True))

final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new file
final_df['label'] = final_df['label'].astype(int)  # Force integer labels
final_df.to_csv(output_file, index=False)
print(f"Balanced dataset saved to {output_file}")

# Scatter plot of examples by class label
plt.figure(figsize=(8, 6))
for label in counter_res.keys():
    row_ix = where(y_resampled == label)[0]
    plt.scatter(X_resampled[row_ix, 0], X_resampled[row_ix, 1],
                label=f"Class {label}", alpha=0.7, edgecolors='k')

plt.xlabel('GT Content')
plt.ylabel('AG Content')
plt.title('Scatter Plot After ADASYN Oversampling')
plt.legend(title='Classes')
plt.grid(alpha=0.5)
plt.show()
