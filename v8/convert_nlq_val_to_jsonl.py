#!/usr/bin/env python3
"""
Convert the original Ego4D NLQ validation set to JSONL format for InternVL3 finetuning.
This will be used as the test set.
"""

import json
import sys
import os

def convert_nlq_val_to_jsonl(input_path, output_path):
    """Convert NLQ validation JSON to JSONL format"""
    
    print(f"Loading NLQ validation data from: {input_path}")
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return False
    
    print(f"Loaded data structure: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        if 'videos' in data:
            print(f"Number of videos: {len(data['videos'])}")
    
    # Extract all language queries from all videos and clips
    all_queries = []
    
    if isinstance(data, dict) and 'videos' in data:
        videos = data['videos']
    elif isinstance(data, list):
        videos = data
    else:
        print(f"Unexpected data structure: {type(data)}")
        return False
    
    for video in videos:
        if not isinstance(video, dict) or 'video_uid' not in video:
            continue
            
        video_uid = video['video_uid']
        
        if 'clips' not in video:
            continue
            
        for clip in video['clips']:
            if not isinstance(clip, dict) or 'clip_uid' not in clip:
                continue
                
            clip_uid = clip['clip_uid']
            
            if 'annotations' not in clip:
                continue
                
            for annotation in clip['annotations']:
                if not isinstance(annotation, dict) or 'language_queries' not in annotation:
                    continue
                    
                for query in annotation['language_queries']:
                    if not isinstance(query, dict) or 'query' not in query:
                        continue
                        
                    # Create a flattened entry for each query
                    entry = {
                        'video_uid': video_uid,
                        'clip_uid': clip_uid,
                        'clip_start_sec': query.get('clip_start_sec', 0),
                        'clip_end_sec': query.get('clip_end_sec', 0),
                        'video_start_sec': query.get('video_start_sec', 0),
                        'video_end_sec': query.get('video_end_sec', 0),
                        'video_start_frame': query.get('video_start_frame', 0),
                        'video_end_frame': query.get('video_end_frame', 0),
                        'query': query['query'],
                        'template': query.get('template', ''),
                        'slot_x': query.get('slot_x', ''),
                        'verb_x': query.get('verb_x', ''),
                        'slot_y': query.get('slot_y', ''),
                        'verb_y': query.get('verb_y', ''),
                        'raw_tags': query.get('raw_tags', [])
                    }
                    all_queries.append(entry)
    
    print(f"Extracted {len(all_queries)} language queries")
    
    # Write to JSONL file
    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        for entry in all_queries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Successfully converted {len(all_queries)} queries to JSONL format")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_nlq_val_to_jsonl.py <input_json> <output_jsonl>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)
    
    success = convert_nlq_val_to_jsonl(input_path, output_path)
    if not success:
        sys.exit(1) 