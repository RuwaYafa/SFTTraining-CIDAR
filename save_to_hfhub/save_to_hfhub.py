# !python save_to_hfhub.py \
#   --model_path "/content/output" \
#   --repo_name "yourusername/AceGPT-CIDAR" \
#   --hf_token "your_hf_token" \
#   --your_name "Your Name" \
#   --private  # Remove this for public repo

# os.makedirs('/content/drive/MyDrive/SFTTraining-CIDAR-main/save_to_hfhub', exist_ok=True) # more pythonic like !mkdir -p /content/output
# !huggingface-cli whoami
## !huggingface-cli repo-info RuwaYafa/AceGPT-CIDAR
# !huggingface-cli login --token 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
# !cd "/content/drive/MyDrive/SFTTraining-CIDAR-main/merge-lora-SFTTraining-CIDAR/" && git lfs install && git add . && git commit -m "/content/drive/MyDrive/SFTTraining-CIDAR-main/merge-lora-SFTTraining-CIDAR/model-00001-of-00007.safetensors" && git push


# !python /content/drive/MyDrive/SFTTraining-CIDAR-main/save_to_hfhub/save_to_hfhub.py \
#   --model_path "/content/drive/MyDrive/SFTTraining-CIDAR-main/merge-lora-SFTTraining-CIDAR/" \
#   --repo_name "RuwaYafa/AceGPT-7B-chat-CIDAR" \
#   --hf_token "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" \
#   --your_name "Ruwa' F. AbuHweidi" \
#   # --private  # Remove this for public repo


#!/usr/bin/env python3
# save_to_hfhub.py - Save fine-tuned AceGPT model to Hugging Face Hub

import os
import logging
from huggingface_hub import HfApi, login, ModelCard
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_to_huggingface(
    model_path: str,
    repo_name: str,
    hf_token: str,
    your_name: str = "Your Name",
    private: bool = False,
    model_card: str = None
):
    """
    Save fine-tuned model to Hugging Face Hub
    
    Args:
        model_path: Path to local model directory
        repo_name: HF repo name (e.g., "yourusername/AceGPT-CIDAR")
        hf_token: Your HF API token
        your_name: Your name as trainer
        private: Make repo private
        model_card: Custom model card (optional)
    """
    try:
        # 1. Authenticate
        login(token=hf_token)
        api = HfApi()
        
        # 2. Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 3. Create repo (if doesn't exist)
        api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True
        )
        
        # 4. Generate model card if not provided
        if not model_card:
            model_card = f"""
---
language:
- ar
- en
license: apache-2.0
tags:
- Arabic
- AceGPT
- CIDAR
- cultural-ai
datasets:
- arbml/CIDAR
pipeline_tag: text-generation
base_model: FreedomIntelligence/AceGPT-7B
model_creator: {your_name}
fine-tuned_by: {your_name}
training_date: {datetime.now().strftime('%Y-%m-%d')}
---

# AceGPT Fine-Tuned on CIDAR

This model is fine-tuned version of AceGPT-7B on the CIDAR dataset for Arabic cultural understanding.

## Training Details
- **Trainer**: {your_name}
- **Dataset**: CIDAR (Arabic cultural Q&A)
- **Base Model**: FreedomIntelligence/AceGPT-7B
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
"""
        
        # 5. Save model card
        with open(os.path.join(model_path, "README.md"), "w") as f:
            f.write(model_card)
        
        # 6. Upload all files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message=f"Fine-tuned AceGPT model by {your_name}",
            repo_type="model"
        )
        
        logger.info(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload model: {e}")
        raise

if __name__ == "__main__":
    # Example usage (for Colab)
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to local model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="HF repo name (e.g., 'yourusername/AceGPT-CIDAR')")
    parser.add_argument("--hf_token", type=str, required=True,
                       help="Your HF API token")
    parser.add_argument("--your_name", type=str, default="Your Name",
                       help="Your name as trainer")
    parser.add_argument("--private", action="store_true",
                       help="Make repo private")
    
    args = parser.parse_args()
    
    save_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hf_token=args.hf_token,
        your_name=args.your_name,
        private=args.private
    )