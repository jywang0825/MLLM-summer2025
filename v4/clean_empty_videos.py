#!/usr/bin/env python3
"""
Clean empty videos from the narrations annotations file.
Removes videos that have no narrations or other issues.
"""

import json
import os

def clean_empty_videos(input_file: str, output_file: str = None):
    """Remove videos with no narrations from the file."""
    if output_file is None:
        output_file = input_file
    
    print(f"📖 Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Total videos before cleaning: {len(data)}")
    
    # Find videos with no narrations
    videos_with_no_narrations = [v for v in data if v.get('total_narrations', 0) == 0]
    print(f"❌ Videos with no narrations: {len(videos_with_no_narrations)}")
    
    # Find videos with errors
    videos_with_errors = [v for v in data if v.get('error')]
    print(f"⚠️  Videos with errors: {len(videos_with_errors)}")
    
    # Remove videos with no narrations or errors
    cleaned_data = [v for v in data if v.get('total_narrations', 0) > 0 and not v.get('error')]
    
    print(f"✅ Videos after cleaning: {len(cleaned_data)}")
    print(f"🗑️  Removed {len(data) - len(cleaned_data)} videos")
    
    # Save cleaned data
    print(f"💾 Saving cleaned data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2, default=str)
    
    print(f"✅ Cleaning completed!")
    
    # Show some stats about the remaining videos
    total_narrations = sum(v.get('total_narrations', 0) for v in cleaned_data)
    avg_narrations = total_narrations / len(cleaned_data) if cleaned_data else 0
    print(f"📈 Average narrations per video: {avg_narrations:.1f}")
    
    return cleaned_data

if __name__ == "__main__":
    input_file = "ego4d_nlq_uniform_frame_narrations.json"
    clean_empty_videos(input_file) 