#!/usr/bin/env python3
"""
Remove the last 4 entries from the ego4d_nlq_uniform_frame_narrations.json file
"""

import json

def remove_last_entries(filename, num_entries=4):
    """Remove the last num_entries from the JSON file."""
    
    # Read the current file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Original file has {len(data)} entries")
    
    # Remove the last num_entries
    if len(data) >= num_entries:
        data = data[:-num_entries]
        print(f"Removed {num_entries} entries")
    else:
        print(f"File has fewer than {num_entries} entries")
        return
    
    print(f"File now has {len(data)} entries")
    
    # Write back to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Updated file saved to {filename}")

if __name__ == "__main__":
    remove_last_entries("ego4d_nlq_uniform_frame_narrations.json", 4) 