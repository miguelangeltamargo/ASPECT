import pandas as pd
import argparse

def split_class(file_path):
    """
    Split the rows with CLASS = 3 into multiple files with up to 10,000 rows each.

    :param file_path: Path to the input file (CSV format).
    """
    try:
        # Load the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)

        # Check if CLASS column exists
        if 'CLASS' not in data.columns:
            print("Error: The file does not contain a 'CLASS' column.")
            return

        # Filter rows where CLASS column value matches specified
        class_two_data = data[data['CLASS'] == 3]

        # Split into chunks of rows
        chunk_size = 8600
        num_chunks = (len(class_two_data) + chunk_size - 1) // chunk_size  # Calculate the number of chunks

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(class_two_data))

            # Extract chunk
            chunk = class_two_data.iloc[start_idx:end_idx]

            # Save chunk to a new file
            output_file = f"Cassette_chunk_{i + 1}.csv"
            chunk.to_csv(output_file, index=False)
            print(f"Saved {len(chunk)} rows to {output_file}")

    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Argument parser for file input
    parser = argparse.ArgumentParser(description="Split rows with CLASS = 3 into multiple CSV files.")
    parser.add_argument("file", type=str, help="Path to the input CSV file")
    args = parser.parse_args()

    # Call the function with the provided file path
    split_class(args.file)
