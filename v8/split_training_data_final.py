#!/usr/bin/env python3
"""
Split current training data into train/validation sets and set up NLQ validation as test set.
This script will:
1. Combine current train+val data
2. Split into 80% train, 20% validation
3. Set up NLQ validation as test set
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict

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

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load current data
    train_file = "internvl3_data/video_summary_dataset_train.jsonl"
    val_file = "internvl3_data/video_summary_dataset_val.jsonl"
    
    print("Loading current data...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    print(f"Current train examples: {len(train_data)}")
    print(f"Current val examples: {len(val_data)}")
    
    # Combine all data
    all_data = train_data + val_data
    print(f"Total combined examples: {len(all_data)}")
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * len(all_data))
    new_train_data = all_data[:split_idx]
    new_val_data = all_data[split_idx:]
    
    print(f"New train examples: {len(new_train_data)}")
    print(f"New validation examples: {len(new_val_data)}")
    
    # Save new splits
    print("Saving new train/validation splits...")
    save_jsonl(new_train_data, "internvl3_data/video_summary_dataset_train_new.jsonl")
    save_jsonl(new_val_data, "internvl3_data/video_summary_dataset_val_new.jsonl")
    
    # Create NLQ validation test set (you'll need to copy this from your mount)
    print("\nSetting up NLQ validation as test set...")
    print("You need to copy your NLQ validation set to:")
    print("internvl3_data/nlq_val_test.jsonl")
    
    # Update meta.json
    meta_config = {
        "train": "video_summary_dataset_train_new.jsonl",
        "validation": "video_summary_dataset_val_new.jsonl", 
        "test": "nlq_val_test.jsonl"
    }
    
    with open("internvl3_data/meta.json", 'w') as f:
        json.dump(meta_config, f, indent=2)
    
    print("\n✅ Data splitting completed!")
    print(f"📊 New splits:")
    print(f"   Train: {len(new_train_data)} examples")
    print(f"   Validation: {len(new_val_data)} examples")
    print(f"   Test: NLQ validation set (copy to nlq_val_test.jsonl)")
    print(f"\n📁 Files created:")
    print(f"   - internvl3_data/video_summary_dataset_train_new.jsonl")
    print(f"   - internvl3_data/video_summary_dataset_val_new.jsonl")
    print(f"   - internvl3_data/meta.json (updated)")
    print(f"\n⚠️  Next step: Copy your NLQ validation set to internvl3_data/nlq_val_test.jsonl")

if __name__ == "__main__":
    main() 