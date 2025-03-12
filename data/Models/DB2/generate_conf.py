import os
import json
from transformers import AutoConfig

def generate_config(checkpoint_dir, base_model="zhihan1996/DNABERT-2-117M"):
    """
    Generate a config.json file for a fine-tuned model.
    
    Args:
        checkpoint_dir (str): Directory where the fine-tuned model is stored.
        base_model (str): Base model name or path used for fine-tuning.
    """
    # Load the base configuration
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    
    # Update the configuration with fine-tuning details
    config.num_labels = 2  # Adjust based on your classification task
    config.use_lora = True  # Example: LoRA usage
    config.lora_r = 8
    config.lora_alpha = 32
    config.lora_dropout = 0.05
    config.task_type = "SEQ_CLS"  # Sequence classification

    # Save the updated config to the checkpoint directory
    config_path = os.path.join(checkpoint_dir, "config.json")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"Generated config.json in {config_path}")


if __name__ == "__main__":
    # Specify your checkpoint directory here
    checkpoint_dir = "./results/DB2_384_Split/model_output/checkpoint-6030"

    # Specify the base model used during fine-tuning
    base_model = "zhihan1996/DNABERT-2-117M"

    # Generate the config file
    generate_config(checkpoint_dir, base_model)