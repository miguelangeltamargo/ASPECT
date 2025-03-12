import pandas as pd
from sklearn.model_selection import train_test_split
import os

SET = [
       "512_Split",
       ]

def split_dataset(input_file, output_dir, test_size=0.4, val_split=0.5, seed=42):
    """
    Splits a dataset into train, validation, and test sets with a 60-20-20 ratio.
/
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_dir (str): Directory to save the train, validation, and test CSV files.
    - test_size (float): Proportion of the dataset to use for test + validation.
    - val_split (float): Proportion of test_size to use for validation.
    - seed (int): Random seed for reproducibility.
    """
    # Load the dataset
    df = pd.read_csv(input_file)
    print("Label 1 Count: " + str(len(df[df['label'] == 1])))
    # Split into train and temp (test + validation)
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=seed
    )
    
    # Split temp into validation and test
    val_df, test_df = train_test_split(
        temp_df, test_size=val_split, stratify=temp_df['label'], random_state=seed
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "dev.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Dataset split complete!")
    print(f"Training set: {len(train_df)} samples -> {train_path}")
    print(f"Validation set: {len(val_df)} samples -> {val_path}")
    print(f"Test set: {len(test_df)} samples -> {test_path}")

    # Print data distribution
    print("\nClass distribution in training set:")
    print("\nClass distribution in validation set:")

    print("\nClass distribution in testing set:")
    
    # Print data distribution after splitting
    print("\nSamples per label in training set:")
    print(train_df['label'].value_counts(normalize=True))
    print(train_df['label'].value_counts())
    print("\nSamples per label in validation set:")
    print(val_df['label'].value_counts(normalize=True))
    print(val_df['label'].value_counts())
    print("\nSamples per label in test set:")
    print(test_df['label'].value_counts(normalize=True))
    print(test_df['label'].value_counts())
    
# Example usage
if __name__ == "__main__":
    for set_name in SET:
        print(f"\n🔄 Splitting dataset: {set_name}")
        input_file = f"../datasets/{set_name}/{set_name}_C_5.csv"  # Path to your input dataset
        output_dir = f"../datasets/{set_name}/split_dataset_5_20_Balanced/"  # Directory to save split datasets
        split_dataset(input_file, output_dir)
        
        input_file = f"../datasets/{set_name}/{set_name}_C_5.csv"  # Path to your input dataset
        output_dir = f"../datasets/{set_name}/split_dataset_5_50_Balanced/"  # Directory to save split datasets
        split_dataset(input_file, output_dir)

        input_file = f"../datasets/{set_name}/{set_name}_C_5.csv"  # Path to your input dataset
        output_dir = f"../datasets/{set_name}/split_dataset_5_80_Balanced/"  # Directory to save split datasets
        split_dataset(input_file, output_dir)
        
        input_file = f"../datasets/{set_name}/{set_name}_C_5.csv"  # Path to your input dataset
        output_dir = f"../datasets/{set_name}/split_dataset_5_100_Balanced/"  # Directory to save split datasets
        split_dataset(input_file, output_dir)
