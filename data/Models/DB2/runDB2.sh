#!/bin/bash


# Step 1: Wait for the first script to finish
echo "Waiting for the first training script to finish..."
while pgrep -f "runDB2Over.sh" > /dev/null; do sleep 1; done
echo "First script completed!"

# Array of datasets in the correct order
datasets=(
  # "150_Split"
  # "256_Split"
  # "384_Split"
  "512_Split"
)

# Corresponding max_length associative array
declare -A datasets_max_length=(
  # ["150_Split"]="512"
  # ["256_Split"]="512"
  # ["384_Split"]="1024"
  ["512_Split"]="1024"
)

# Base output directory
BASE_OUTPUT_DIR="./results"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
  max_length=${datasets_max_length[$dataset]}
  
  echo "=============================="
  echo "Starting training for dataset: $dataset with max_length: $max_length"
  echo "=============================="
  
  # Define a unique run name
  run_name="DB2_Reduced_Inclusion_100_Alt5"
  # run_name="DB2_${dataset}_Alt5"
  
  # Define a unique output directory
  output_dir="${BASE_OUTPUT_DIR}/${run_name}"
  mkdir -p "$output_dir"
  
  # Determine batch size based on max_length to prevent OOM
  if [ "$max_length" -le 512 ]; then
    train_batch_size=32
    eval_batch_size=32
  elif [ "$max_length" -le 1024 ]; then
    train_batch_size=16
    eval_batch_size=32
    else
    train_batch_size=16
    eval_batch_size=32
  fi

  echo "Using train_batch_size=$train_batch_size and eval_batch_size=$eval_batch_size"
  
  # Run the training script with appropriate parameters
  python ASPECT2_Alt5.py \
    --data_path "./datasets/${dataset}/TeamsShare/Reduced_Inclusion_100/cons_8000_alt5/" \
    --model_name_or_path "zhihan1996/DNABERT-2-117M" \
    --use_wandb \
    --use_lora \
    --use_class_weights \
    --use_optuna \
    --model_max_length "$max_length" \
    --per_device_train_batch_size "$train_batch_size" \
    --per_device_eval_batch_size "$eval_batch_size" \
    --fp16 \
    --gradient_accumulation_steps 4 \
    --optuna_trials 10 \
    --output_dir "$output_dir" \
    --run_name "$run_name" | tee "${output_dir}/${dataset}_training.log"
  
  echo "Finished training for dataset: $dataset"
  echo ""
done
