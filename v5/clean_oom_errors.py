#!/usr/bin/env python3
"""
Clean CUDA out of memory errors from the ego4d captioning results.
"""
import json
import os

def clean_oom_errors(input_file, output_file=None):
    """Remove entries with CUDA out of memory errors."""
    if output_file is None:
        output_file = input_file.replace('.json', '_cleaned.json')
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Original data has {len(data)} entries")
    
    # Filter out entries with CUDA OOM errors
    cleaned_data = []
    oom_count = 0
    
    for entry in data:
        caption = str(entry.get('generated_caption', ''))
        if 'cuda out of memory' in caption.lower():
            oom_count += 1
            print(f"Removing OOM entry: {entry.get('video_uid', 'unknown')}")
        else:
            cleaned_data.append(entry)
    
    print(f"Removed {oom_count} entries with CUDA OOM errors")
    print(f"Cleaned data has {len(cleaned_data)} entries")
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned data saved to {output_file}")
    
    # Show some statistics
    successful_captions = [d for d in cleaned_data if d.get('generated_caption') and not d['generated_caption'].startswith('Error:')]
    print(f"Successful captions: {len(successful_captions)}")
    print(f"Failed captions (other errors): {len(cleaned_data) - len(successful_captions)}")
    
    return cleaned_data

if __name__ == "__main__":
    input_file = "ego4d_uniform_captions_llava_video_7b_qwen2.json"
    cleaned_data = clean_oom_errors(input_file)
    
    # Optionally replace the original file
    print("\nDo you want to replace the original file with the cleaned version? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        os.rename(input_file, input_file + '.backup')
        os.rename(input_file.replace('.json', '_cleaned.json'), input_file)
        print(f"Original file backed up to {input_file}.backup")
        print(f"Cleaned file saved as {input_file}")
    else:
        print("Original file preserved. Cleaned file saved as ego4d_uniform_captions_llava_video_7b_qwen2_cleaned.json") 