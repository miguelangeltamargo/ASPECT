# Corresponding max_length associative array
declare -A datasets_max_length=(
  ["384_Split"]="1024"
)

# Base output directory
BASE_OUTPUT_DIR="./results"

# Loop through each dataset
for dataset in "${!datasets_max_length[@]}"; do
  max_length=${datasets_max_length[$dataset]}
  
  echo "=============================="
  echo "Starting evaluation for dataset: $dataset with max_length: $max_length"
  echo "=============================="
  
  # Define the unique run name
  run_name="DB2_${dataset}"
  
  # Define the output directory where the model checkpoint is stored
  output_dir="${BASE_OUTPUT_DIR}/${run_name}/model_output"
  results_dir="${BASE_OUTPUT_DIR}/${run_name}"
  
  if [ ! -d "$output_dir" ]; then
    echo "Checkpoint directory not found: $output_dir"
    echo "Skipping evaluation for dataset: $dataset"
    continue
  fi
  
  # Define the test dataset path
  test_data_path="./datasets/${dataset}/split_dataset/"
  
  # Run the evaluation script with appropriate parameters
  python evaluate_DB2_384_Split.py \
    --data_path "$test_data_path" \
    --model_name_or_path "$output_dir" \
    --model_max_length "$max_length" \
    --per_device_eval_batch_size 32 \
    --output_dir "$output_dir" | tee "${results_dir}/${dataset}_evaluation.log"
  
  echo "Finished evaluation for dataset: $dataset"
  echo ""
done