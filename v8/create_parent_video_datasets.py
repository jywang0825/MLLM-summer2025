#!/usr/bin/env python3
"""
Create InternVL3 datasets that use parent videos instead of individual clips.
This maps NLQ annotations to the parent video level for full video training.
"""

import json
import csv
import os
import random
from collections import defaultdict

def load_clips_manifest(manifest_path):
    """Load clips manifest and create parent_video_uid -> s3_path mapping"""
    parent_video_paths = {}
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent_video_uid = row['parent_video_uid']
            s3_path = row['s3_path']
            
            # Store the first clip path for each parent video
            # (we'll use this as a reference to construct parent video path)
            if parent_video_uid not in parent_video_paths:
                parent_video_paths[parent_video_uid] = s3_path
    
    return parent_video_paths

def load_nlq_data(nlq_path):
    """Load NLQ data and extract video-level annotations"""
    with open(nlq_path, 'r') as f:
        data = json.load(f)
    
    # Group annotations by parent video
    video_annotations = defaultdict(list)
    
    for video in data['videos']:
        video_uid = video['video_uid']
        
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            
            for annotation in clip['annotations']:
                for query in annotation['language_queries']:
                    # Skip if query field is missing
                    if 'query' not in query or not query['query']:
                        continue
                        
                    # Create video-level entry
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
                    video_annotations[video_uid].append(entry)
    
    return video_annotations

def create_parent_video_dataset(nlq_data, parent_video_paths, output_path):
    """Create JSONL dataset with parent video references"""
    
    all_entries = []
    
    for video_uid, annotations in nlq_data.items():
        if video_uid not in parent_video_paths:
            print(f"Warning: No clip path found for parent video {video_uid}")
            continue
            
        # Get reference clip path to construct parent video path
        clip_path = parent_video_paths[video_uid]
        
        # Construct parent video path (remove clip-specific parts)
        # Example: s3://ego4d-cmu/public/v2/clips/00030ae8-e5c4-41ec-a8ea-63548a7665d6.mp4
        # Parent video might be: s3://ego4d-cmu/public/v2/videos/{parent_video_uid}.mp4
        path_parts = clip_path.split('/')
        if len(path_parts) >= 6:
            # Replace 'clips' with 'videos' and use parent_video_uid
            path_parts[-2] = 'videos'
            path_parts[-1] = f"{video_uid}.mp4"
            parent_video_path = '/'.join(path_parts)
        else:
            parent_video_path = clip_path  # Fallback
        
        for annotation in annotations:
            entry = {
                'video_uid': video_uid,
                'video_path': parent_video_path,
                'clip_uid': annotation['clip_uid'],
                'clip_start_sec': annotation['clip_start_sec'],
                'clip_end_sec': annotation['clip_end_sec'],
                'video_start_sec': annotation['video_start_sec'],
                'video_end_sec': annotation['video_end_sec'],
                'video_start_frame': annotation['video_start_frame'],
                'video_end_frame': annotation['video_end_frame'],
                'query': annotation['query'],
                'template': annotation['template'],
                'slot_x': annotation['slot_x'],
                'verb_x': annotation['verb_x'],
                'slot_y': annotation['slot_y'],
                'verb_y': annotation['verb_y'],
                'raw_tags': annotation['raw_tags']
            }
            all_entries.append(entry)
    
    # Write to JSONL
    with open(output_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    
    return len(all_entries)

def create_test_set_from_val(nlq_val_path, parent_video_paths, output_path):
    """Create a test set using only the parent videos from the original NLQ validation set."""
    val_data = load_nlq_data(nlq_val_path)
    all_entries = []
    for video_uid, annotations in val_data.items():
        if video_uid not in parent_video_paths:
            print(f"Warning: No clip path found for parent video {video_uid}")
            continue
        clip_path = parent_video_paths[video_uid]
        path_parts = clip_path.split('/')
        if len(path_parts) >= 6:
            path_parts[-2] = 'videos'
            path_parts[-1] = f"{video_uid}.mp4"
            parent_video_path = '/'.join(path_parts)
        else:
            parent_video_path = clip_path
        for annotation in annotations:
            entry = {
                'video_uid': video_uid,
                'video_path': parent_video_path,
                'clip_uid': annotation['clip_uid'],
                'clip_start_sec': annotation['clip_start_sec'],
                'clip_end_sec': annotation['clip_end_sec'],
                'video_start_sec': annotation['video_start_sec'],
                'video_end_sec': annotation['video_end_sec'],
                'video_start_frame': annotation['video_start_frame'],
                'video_end_frame': annotation['video_end_frame'],
                'query': annotation['query'],
                'template': annotation['template'],
                'slot_x': annotation['slot_x'],
                'verb_x': annotation['verb_x'],
                'slot_y': annotation['slot_y'],
                'verb_y': annotation['verb_y'],
                'raw_tags': annotation['raw_tags']
            }
            all_entries.append(entry)
    with open(output_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Created test set with {len(all_entries)} entries from original NLQ val videos.")
    return len(all_entries)

def get_unique_videos(nlq_path):
    with open(nlq_path, 'r') as f:
        data = json.load(f)
    unique_videos = set()
    for video in data['videos']:
        video_uid = video['video_uid']
        unique_videos.add(video_uid)
    return sorted(unique_videos)

def write_unique_video_jsonl(video_uids, parent_video_paths, output_path):
    all_entries = []
    for video_uid in video_uids:
        if video_uid not in parent_video_paths:
            print(f"Warning: No clip path found for parent video {video_uid}")
            continue
        clip_path = parent_video_paths[video_uid]
        path_parts = clip_path.split('/')
        if len(path_parts) >= 6:
            path_parts[-2] = 'videos'
            path_parts[-1] = f"{video_uid}.mp4"
            parent_video_path = '/'.join(path_parts)
        else:
            parent_video_path = clip_path
        entry = {
            'video_uid': video_uid,
            'video_path': parent_video_path
        }
        all_entries.append(entry)
    with open(output_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Wrote {len(all_entries)} entries to {output_path}")
    return len(all_entries)

def main():
    random.seed(42)
    clips_manifest = "/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/clips/manifest.csv"
    nlq_train = "/shared/ssd_14T/home/wangj/your-repo/finetuning/remote_ego4d/v2/annotations/nlq_train.json"
    output_dir = "internvl3_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading clips manifest...")
    parent_video_paths = load_clips_manifest(clips_manifest)
    print(f"Found {len(parent_video_paths)} parent videos")
    
    # 1. Get unique video_uids from train set and split 80/20
    train_videos = get_unique_videos(nlq_train)
    random.shuffle(train_videos)
    split_idx = int(0.8 * len(train_videos))
    train_split = train_videos[:split_idx]
    val_split = train_videos[split_idx:]
    
    # 2. Write JSONL files for train and validation only
    train_count = write_unique_video_jsonl(train_split, parent_video_paths, f"{output_dir}/nlq_train_parent_videos_unique.jsonl")
    val_count = write_unique_video_jsonl(val_split, parent_video_paths, f"{output_dir}/nlq_val_parent_videos_unique.jsonl")
    
    # 3. Update meta.json (keeping existing test set reference)
    meta = {
        "train": "nlq_train_parent_videos_unique.jsonl",
        "validation": "nlq_val_parent_videos_unique.jsonl",
        "test": "nlq_test_parent_videos_unique.jsonl"  # Keep existing test set
    }
    with open(f"{output_dir}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("Created train and validation sets from 80/20 split of NLQ train videos.")
    print(f"Train: {train_count} unique videos")
    print(f"Validation: {val_count} unique videos")
    print("Test set unchanged (using existing nlq_test_parent_videos_unique.jsonl)")

if __name__ == "__main__":
    main() 