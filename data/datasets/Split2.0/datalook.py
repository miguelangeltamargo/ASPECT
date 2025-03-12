import pandas as pd
import argparse

def count_class_one(file_path):
    """
    Count the number of rows with the value 1 in the CLASS column.
    
    :param file_path: Path to the input file (CSV format).
    """
    try:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        
        # Check if CLASS column exists
        if 'CLASS' not in data.columns:
            print("Error: The file does not contain a 'CLASS' column.")
            return
        
        # Count rows where CLASS column has value 1
        count = data[data['CLASS'] == 1].shape[0]
        print(f"Number of rows with CLASS = 1: {count}")
        
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Argument parser for file input
    parser = argparse.ArgumentParser(description="Count rows with CLASS = 1 in a CSV file.")
    parser.add_argument("file", type=str, help="Path to the input CSV file")
    args = parser.parse_args()
    
    # Call the function with the provided file path
    count_class_one(args.file)