#!/usr/bin/env python3
"""
Clean up ego4d_aks_captions.json to keep only entries with generated_summary filled out
"""
import json
import os

def cleanup_ego4d_captions():
    """Clean up ego4d_aks_captions.json to keep only entries with generated_summary."""
    
    input_file = "ego4d_aks_captions.json"
    backup_file = "ego4d_aks_captions.json.backup"
    output_file = "ego4d_aks_captions_cleaned.json"
    
    print(f"Loading data from {input_file}...")
    
    # Load the original data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original entries: {len(data)}")
    
    # Create backup
    print(f"Creating backup: {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Filter entries with generated_summary
    cleaned_data = []
    for entry in data:
        generated_summary = entry.get('generated_summary', '')
        if generated_summary and generated_summary.strip():
            cleaned_data.append(entry)
    
    print(f"Entries with generated_summary: {len(cleaned_data)}")
    print(f"Removed {len(data) - len(cleaned_data)} entries without generated_summary")
    
    # Save cleaned data
    print(f"Saving cleaned data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Optionally replace the original file
    print(f"Replacing original file with cleaned version...")
    os.rename(output_file, input_file)
    
    print("Cleanup completed!")
    print(f"Original backup saved as: {backup_file}")
    print(f"Cleaned file: {input_file}")

if __name__ == "__main__":
    cleanup_ego4d_captions() 