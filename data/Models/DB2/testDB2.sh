#!/bin/bash


# Array of datasets in the correct order
datasets=(
  # "150_Split"
  # "256_Split"
  # "384_Split"
  "512_Split"
)

# Base output directory
BASE_OUTPUT_DIR="./results"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  
  echo "================================================="
  echo "Starting Evaluation for dataset: $dataset"
  echo "================================================="
  
  # Define a unique run name
  run_name="DB2_${dataset}_Alt5"
  
  TESTING_MODEL_PATH="./results/DB2_512_Split_Alt5_LastRun/model_output/"
  # Define a unique output directory
  output_dir="${BASE_OUTPUT_DIR}/${run_name}/evaluation_results_sahil_synthetic"
  mkdir -p "$output_dir"
  
    python Test_The_Best.py \
        --model_path "$TESTING_MODEL_PATH" \
        --test_data_path "./datasets/512_Split/split_dataset_5_US/sahil_synthetic_test.csv" \
        --output_dir "$output_dir" \
        --batch_size 32 \
        --kmer -1

          # Define a unique output directory
  output_dir="${BASE_OUTPUT_DIR}/${run_name}/evaluation_results_undersample"
  mkdir -p "$output_dir"
  
    python Test_The_Best.py \
        --model_path "$TESTING_MODEL_PATH" \
        --test_data_path "./datasets/512_Split/split_dataset_5_US/train.csv" \
        --output_dir "$output_dir" \
        --batch_size 32 \
        --kmer -1

  output_dir="${BASE_OUTPUT_DIR}/${run_name}/evaluation_results_20_balanced"
  mkdir -p "$output_dir"

    python Test_The_Best.py \
        --model_path "$TESTING_MODEL_PATH" \
        --test_data_path "./datasets/512_Split/split_dataset_5_20_Balanced/test.csv" \
        --output_dir "$output_dir" \
        --batch_size 32 \
        --kmer -1

  output_dir="${BASE_OUTPUT_DIR}/${run_name}/evaluation_results_50_balanced"
  mkdir -p "$output_dir"

    python Test_The_Best.py \
        --model_path "$TESTING_MODEL_PATH" \
        --test_data_path "./datasets/512_Split/split_dataset_5_50_Balanced/test.csv" \
        --output_dir "$output_dir" \
        --batch_size 32 \
        --kmer -1

done