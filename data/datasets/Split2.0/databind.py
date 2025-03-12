import pandas as pd
import argparse

def combine_and_modify_classes(file1, file2, new_class, output_file):
    """
    Combine two CSV files, modify the CLASS column to a new value, and save to a new file.

    :param file1: Path to the first input CSV file.
    :param file2: Path to the second input CSV file.
    :param new_class: New class value to assign to all rows.
    :param output_file: Path to the output CSV file.
    """
    try:
        # Load the CSV files into pandas DataFrames
        data1 = pd.read_csv(file1)
        data2 = pd.read_csv(file2)

        # Modify the CLASS column to the new value
        if 'CLASS' in data2.columns:
            data2['CLASS'] = new_class
        else:
            print("Error: One or both files do not contain a 'CLASS' column.")
            return

        # Combine the DataFrames
        combined_data = pd.concat([data1, data2], ignore_index=True)

        # Shuffle combined DataFrames
        combined_shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save the combined DataFrame to a new file
        combined_shuffled_data.to_csv(output_file, index=False)
        print(f"Combined data saved to {output_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Argument parser for file input
    parser = argparse.ArgumentParser(description="Combine two files and modify the CLASS column.")
    parser.add_argument("file1", type=str, help="Path to the first input CSV file")
    parser.add_argument("file2", type=str, help="Path to the second input CSV file")
    parser.add_argument("new_class", type=int, help="New class value to assign to all rows")
    parser.add_argument("output", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    # Call the function to combine files and modify classes
    combine_and_modify_classes(args.file1, args.file2, args.new_class, args.output)
