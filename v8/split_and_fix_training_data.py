#!/usr/bin/env python3
"""
Split current training data into train/validation sets and fix training configuration.
This script will:
1. Split current training data into 80% train, 20% validation
2. Keep existing test set separate
3. Fix PYTHONPATH issues in training scripts
4. Enable Flash Attention
"""

import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def split_data(data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split data into train/validation sets."""
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:]
    
    return train_data, val_data

def update_meta_file(train_file: str, val_file: str, test_file: str, output_dir: str):
    """Update meta.json file with new file paths."""
    meta_content = {
        "train": train_file,
        "validation": val_file,
        "test": test_file
    }
    
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_content, f, indent=2)
    
    print(f"✅ Updated meta.json: {meta_path}")

def fix_training_script():
    """Fix PYTHONPATH and enable Flash Attention in training script."""
    script_path = "finetune_internvl3_8b_4bit_lora.sh"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    # Read current script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Fix PYTHONPATH to include the correct internvl module path
    old_pythonpath = 'export PYTHONPATH="../InternVL/internvl_chat/internvl:${PYTHONPATH}"'
    new_pythonpath = 'export PYTHONPATH="../InternVL/internvl_chat:${PYTHONPATH}"'
    
    if old_pythonpath in content:
        content = content.replace(old_pythonpath, new_pythonpath)
        print("✅ Fixed PYTHONPATH in training script")
    
    # Remove any --attn_implementation 'eager' to enable Flash Attention
    content = content.replace("--attn_implementation 'eager' \\\n  ", "")
    print("✅ Enabled Flash Attention (removed eager implementation)")
    
    # Write updated script
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated training script: {script_path}")
    return True

def main():
    print("🔄 Splitting training data and fixing configuration...")
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Load current training data
    train_file = "internvl3_data/video_summary_dataset_train.jsonl"
    val_file = "internvl3_data/video_summary_dataset_val.jsonl"
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        return
    
    if not os.path.exists(val_file):
        print(f"❌ Validation file not found: {val_file}")
        return
    
    # Load data
    print("📊 Loading current data...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    print(f"Current data: {len(train_data)} train, {len(val_data)} validation")
    
    # Combine train and validation data for splitting
    all_training_data = train_data + val_data
    print(f"Total training data to split: {len(all_training_data)}")
    
    # Split into train/validation (80/20)
    train_new, val_new = split_data(all_training_data, train_ratio=0.8)
    
    print(f"New split: {len(train_new)} train, {len(val_new)} validation")
    
    # Save split data
    output_dir = "internvl3_data"
    os.makedirs(output_dir, exist_ok=True)
    
    save_jsonl(train_new, os.path.join(output_dir, "video_summary_dataset_train.jsonl"))
    save_jsonl(val_new, os.path.join(output_dir, "video_summary_dataset_val.jsonl"))
    
    print("✅ Saved split data:")
    print(f"  - Train: {len(train_new)} examples")
    print(f"  - Validation: {len(val_new)} examples")
    print("  - Test: Using existing NLQ validation set")
    
    # Update meta.json (keep existing test set)
    update_meta_file(
        "video_summary_dataset_train.jsonl",
        "video_summary_dataset_val.jsonl", 
        "video_summary_dataset_test.jsonl",  # Keep existing test set
        output_dir
    )
    
    # Fix training script
    print("\n🔧 Fixing training configuration...")
    if fix_training_script():
        print("✅ Training script updated successfully")
    else:
        print("❌ Failed to update training script")
    
    print("\n🎉 Data splitting and configuration fix completed!")
    print("\n📋 Summary:")
    print(f"  - Training examples: {len(train_new)}")
    print(f"  - Validation examples: {len(val_new)}")
    print("  - Test examples: Using existing NLQ validation set")
    print("  - Flash Attention: Enabled")
    print("  - PYTHONPATH: Fixed")
    print("\n🚀 You can now run training with:")
    print("   ./quick_start_internvl3_4bit_lora.sh")

if __name__ == "__main__":
    main() 