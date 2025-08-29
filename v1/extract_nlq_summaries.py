#!/usr/bin/env python3
"""
Extract video summaries for NLQ train and test videos from Ego4D narrations.
This script uses the manifest.csv file to map clip UIDs to parent video UIDs
and extracts summaries only for videos that appear in NLQ annotations.
"""

import json
import os
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm

def load_manifest(manifest_path: str) -> Dict[str, str]:
    """Load clips manifest and create mapping from clip_uid to parent_video_uid."""
    print(f"Loading clips manifest from {manifest_path}")
    clip_to_parent = {}
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exported_clip_uid = row['exported_clip_uid']
            parent_video_uid = row['parent_video_uid']
            clip_to_parent[exported_clip_uid] = parent_video_uid
    
    print(f"Loaded {len(clip_to_parent)} clip-to-parent mappings")
    return clip_to_parent

def load_nlq_annotations(nlq_path: str) -> Set[str]:
    """Load NLQ annotations and extract clip UIDs from the 'videos' key."""
    print(f"Loading NLQ annotations from {nlq_path}")
    
    with open(nlq_path, 'r') as f:
        nlq_data = json.load(f)
    
    print(f"Type of nlq_data: {type(nlq_data)}")
    print(f"Keys in nlq_data: {list(nlq_data.keys()) if isinstance(nlq_data, dict) else 'Not a dict'}")
    
    clip_uids = set()
    
    # The structure is: {"videos": [{"video_uid": "...", "clips": [{"clip_uid": "...", ...}, ...]}, ...]}
    if isinstance(nlq_data, dict) and 'videos' in nlq_data:
        videos = nlq_data['videos']
        print(f"Found {len(videos)} videos in NLQ annotation file")
        for video_entry in videos:
            if isinstance(video_entry, dict) and 'clips' in video_entry:
                clips = video_entry['clips']
                for clip in clips:
                    if isinstance(clip, dict) and 'clip_uid' in clip:
                        clip_uid = clip['clip_uid']
                        clip_uids.add(clip_uid)
    
    print(f"Found {len(clip_uids)} unique clip UIDs in NLQ annotations")
    return clip_uids

def load_narrations(narration_path: str) -> Dict:
    """Load Ego4D narrations from JSON file."""
    print(f"Loading narrations from {narration_path}")
    with open(narration_path, 'r') as f:
        return json.load(f)

def extract_video_summaries_for_nlq(narrations: Dict, nlq_clip_uids: Set[str], 
                                   clip_to_parent: Dict[str, str], output_dir: str) -> Dict:
    """Extract video summaries for NLQ videos from Ego4D narrations."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    summaries_data = {}
    found_videos = 0
    
    print(f"Processing narrations for {len(nlq_clip_uids)} NLQ clips...")
    
    # First, map NLQ clip UIDs to parent video UIDs
    nlq_parent_videos = set()
    for clip_uid in nlq_clip_uids:
        if clip_uid in clip_to_parent:
            parent_video_uid = clip_to_parent[clip_uid]
            nlq_parent_videos.add(parent_video_uid)
        else:
            print(f"Warning: Clip UID {clip_uid} not found in manifest")
    
    print(f"Found {len(nlq_parent_videos)} parent videos for NLQ clips")
    
    # Process each video in narrations
    for video_uid, video_data in tqdm(narrations.items()):
        if video_uid in nlq_parent_videos:
            found_videos += 1
            
            # Extract summary from the video data
            summary = None
            if 'narration_pass_1' in video_data:
                narration_pass = video_data['narration_pass_1']
                if 'summaries' in narration_pass:
                    summaries = narration_pass['summaries']
                    if isinstance(summaries, list) and len(summaries) > 0:
                        # Take the first summary and extract the text
                        first_summary = summaries[0]
                        if 'summary_text' in first_summary:
                            summary = first_summary['summary_text']
                    elif isinstance(summaries, dict):
                        if 'summary_text' in summaries:
                            summary = summaries['summary_text']
            
            if summary:
                summaries_data[video_uid] = {
                    'video_uid': video_uid,
                    'summary': summary,
                    'video_data': video_data
                }
    
    print(f"Found summaries for {len(summaries_data)} out of {found_videos} NLQ videos")
    
    # Save summaries to JSON file
    output_file = os.path.join(output_dir, 'nlq_video_summaries.json')
    with open(output_file, 'w') as f:
        json.dump(summaries_data, f, indent=2)
    
    print(f"Saved summaries to {output_file}")
    
    return summaries_data

def create_nlq_format(summaries_data: Dict, output_dir: str) -> None:
    """Create NLQ format dataset with video summaries."""
    
    nlq_format_data = []
    
    for video_uid, video_info in summaries_data.items():
        nlq_format_data.append({
            'video_uid': video_uid,
            'summary': video_info['summary'],
            'video_data': video_info['video_data']
        })
    
    # Save in NLQ format
    output_file = os.path.join(output_dir, 'nlq_summaries_dataset.json')
    with open(output_file, 'w') as f:
        json.dump(nlq_format_data, f, indent=2)
    
    print(f"Created NLQ format dataset with {len(nlq_format_data)} videos")
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract video summaries for NLQ videos')
    parser.add_argument('--ego4d_path', default='~/remote_ego4d/v2', 
                       help='Path to Ego4D v2 dataset')
    parser.add_argument('--output_dir', default='./nlq_summaries', 
                       help='Output directory for extracted summaries')
    parser.add_argument('--create_nlq_format', action='store_true',
                       help='Create NLQ format dataset')
    
    args = parser.parse_args()
    
    # Expand user path
    ego4d_path = os.path.expanduser(args.ego4d_path)
    
    # File paths
    manifest_path = os.path.join(ego4d_path, 'clips', 'manifest.csv')
    nlq_train_path = os.path.join(ego4d_path, 'annotations', 'nlq_train.json')
    nlq_test_path = os.path.join(ego4d_path, 'annotations', 'nlq_val.json')
    narration_path = os.path.join(ego4d_path, 'annotations', 'narration.json')
    
    # Check if files exist
    for path in [manifest_path, nlq_train_path, nlq_test_path, narration_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return
    
    # Load data
    clip_to_parent = load_manifest(manifest_path)
    nlq_train_clips = load_nlq_annotations(nlq_train_path)
    nlq_test_clips = load_nlq_annotations(nlq_test_path)
    narrations = load_narrations(narration_path)
    
    # Combine train and test clip UIDs
    all_nlq_clips = nlq_train_clips.union(nlq_test_clips)
    print(f"Total unique NLQ clips: {len(all_nlq_clips)}")
    
    # Extract summaries for NLQ clips
    summaries_data = extract_video_summaries_for_nlq(narrations, all_nlq_clips, clip_to_parent, args.output_dir)
    
    # Create NLQ format if requested
    if args.create_nlq_format:
        create_nlq_format(summaries_data, args.output_dir)

if __name__ == '__main__':
    main() 