#!/bin/bash

# Exit the script immediately if any command fails
set -e

# Directory containing all dataset CSV files
DATASET_DIR="../../datasets/sahil/new_data_set/CSV/"

# Check if the dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory $DATASET_DIR does not exist."
    exit 1
fi

# Iterate through all CSV files in the directory
for dataset_file in "$DATASET_DIR"*.csv; do
    if [ -f "$dataset_file" ]; then
        echo "=== Running experiment for $dataset_file ==="
        # Run the Python script and capture any errors
        python3 ASP24_QTesting.py --dataset_path "$dataset_file"
        # Check the exit status of the Python script
        if [ $? -ne 0 ]; then
            echo "Error encountered while processing $dataset_file. Stopping script."
            exit 1
        fi
    else
        echo "No CSV files found in $DATASET_DIR"
    fi
done

echo "=== All experiments completed successfully! ==="
