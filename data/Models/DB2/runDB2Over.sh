#!/bin/bash

# Step 1: Wait for the first script to finish
echo "Waiting for the first training script to finish..."
while pgrep -f "runDB2.sh" > /dev/null; do sleep 1; done
echo "First script completed!"

# # Step 2: Rerun the 1024_Split dataset
# echo "Rerunning training for dataset: 1024_Split"
# python ASPECT2.py \
#   --data_path "./datasets/1024_Split/split_dataset/" \
#   --model_name_or_path "zhihan1996/DNABERT-2-117M" \
#   --use_wandb \
#   --use_lora \
#   --model_max_length 2048 \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 8 \
#   --num_train_epochs 10 \
#   --learning_rate 3e-5 \
#   --seed 42 \
#   --fp16 \
#   --gradient_accumulation_steps 4 \
#   --output_dir "./results/DB2_1024_Split" \
#   --run_name "DB2_1024_Split" | tee "./results/DB2_1024_Split/training.log"
# echo "Finished rerunning training for dataset: 1024_Split"

# Step 3: Run all 5 datasets with oversampling enabled
echo "Starting training with oversampling for all datasets"

# Oversampling Bash Loop Script
declare -A datasets_max_length=(
  ["75_Split"]="512"
  ["128_Split"]="512"
  ["256_Split"]="512"
  ["512_Split"]="1024"
  # ["1024_Split"]="2048"
)

for dataset in "${!datasets_max_length[@]}"
do
  max_length=${datasets_max_length[$dataset]}
  
  echo "=============================="
  echo "Starting oversampling training for dataset: $dataset with max_length: $max_length"
  echo "=============================="
  
  # Define a unique run name
  run_name="DB2_${dataset}_Oversampled"
  
  # Define a unique output directory
  output_dir="./results/${run_name}"
  mkdir -p "$output_dir"
  
  # Determine batch size based on max_length to prevent OOM
  if [ "$max_length" -le 512 ]; then
    train_batch_size=32
    eval_batch_size=32
  elif [ "$max_length" -le 1024 ]; then
    train_batch_size=16
    eval_batch_size=32
    else
    train_batch_size=8
    eval_batch_size=8
  fi

  echo "Using train_batch_size=$train_batch_size and eval_batch_size=$eval_batch_size"

  # Run the training script with oversampling enabled
  python ASPECT2.py \
    --data_path "./datasets/${dataset}/split_dataset/" \
    --model_name_or_path "zhihan1996/DNABERT-2-117M" \
    --use_wandb \
    --use_lora \
    --apply_adasyn \
    --model_max_length "$max_length" \
    --per_device_train_batch_size "$train_batch_size" \
    --per_device_eval_batch_size "$eval_batch_size" \
    --num_train_epochs 10 \
    --learning_rate 3e-5 \
    --seed 42 \
    --fp16 \
    --gradient_accumulation_steps 4 \
    --output_dir "$output_dir" \
    --run_name "$run_name" | tee "${output_dir}/training.log"

  echo "Finished oversampling training for dataset: $dataset"
  echo ""
done

echo "All oversampling training completed!"
