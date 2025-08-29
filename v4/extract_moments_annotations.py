#!/usr/bin/env python3
"""
Ego4D Moments Annotations Extractor for NLQ Validation Dataset
Extracts moments annotations for all videos in the NLQ validation set.
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import traceback
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse

def load_ego4d_annotations(annotation_path: str) -> Dict:
    """Load Ego4D annotations from JSON file."""
    print(f"Loading annotations from {annotation_path}")
    with open(annotation_path, 'r') as f:
        return json.load(f)

def load_nlq_val_videos(summaries_path: str = "../v1/nlq_val_summaries.json", 
                       nlq_val_path: str = "../remote_ego4d/v2/annotations/nlq_val.json") -> List[Dict]:
    """Load video UIDs from NLQ summaries and find their clips."""
    try:
        # Load summaries first (this is our source of truth)
        with open(summaries_path, 'r') as f:
            summaries_data = json.load(f)
        
        # Load NLQ validation data to get clip info
        with open(nlq_val_path, 'r') as f:
            nlq_data = json.load(f)
        
        # Create mapping from video_uid to clips
        video_clips_map = {}
        for video in nlq_data['videos']:
            if video['clips']:
                video_clips_map[video['video_uid']] = video['clips'][0]['clip_uid']
        
        video_data = []
        for summary_item in summaries_data:
            video_uid = summary_item['video_uid']
            original_summary = summary_item['summary']
            
            if video_uid in video_clips_map:
                # Video has clips - use the first clip
                video_data.append({
                    'video_uid': video_uid,
                    'clip_uid': video_clips_map[video_uid],
                    'original_summary': original_summary,
                    'has_clips': True
                })
            else:
                # Video has no clips - mark as not processable
                video_data.append({
                    'video_uid': video_uid,
                    'clip_uid': None,
                    'original_summary': original_summary,
                    'has_clips': False
                })
        
        print(f"Loaded {len(video_data)} videos from NLQ summaries")
        print(f"Videos with clips: {len([v for v in video_data if v['has_clips']])}")
        print(f"Videos without clips: {len([v for v in video_data if not v['has_clips']])}")
        return video_data
    except Exception as e:
        print(f"Error loading NLQ validation data: {e}")
        return []

def find_video_path(clip_uid: str, clips_manifest_path: str = "../remote_ego4d/v2/clips/manifest.csv") -> Optional[str]:
    """Find the video file path for a given clip UID from the manifest."""
    try:
        df = pd.read_csv(clips_manifest_path)
        # Look for the clip UID in the exported_clip_uid column
        match = df[df['exported_clip_uid'] == clip_uid]
        if not match.empty:
            # Use the manifold_location path
            manifold_path = match.iloc[0]['manifold_location']
            # Convert manifold path to local path
            local_path = manifold_path.replace('manifold://ego4d_fair/tree/exported_clips/', '../remote_ego4d/v2/clips/')
            return local_path
        else:
            print(f"Clip UID {clip_uid} not found in manifest")
            return None
    except Exception as e:
        print(f"Error finding video path for {clip_uid}: {e}")
        return None

def extract_moments_annotations(moments_data: Dict, clip_uid: str) -> List[Dict]:
    """Extract moments annotations for the given clip."""
    moments_annotations = []
    
    # Find the clip in moments data
    for video in moments_data.get('videos', []):
        for clip in video.get('clips', []):
            if clip.get('clip_uid') == clip_uid:
                # Get annotations for this clip
                for annotation in clip.get('annotations', []):
                    # Extract moment information
                    moment_info = {
                        'annotation_uid': annotation.get('annotation_uid', ''),
                        'moment_start_sec': annotation.get('moment_start_sec', 0),
                        'moment_end_sec': annotation.get('moment_end_sec', 0),
                        'moment_start_frame': annotation.get('moment_start_frame', 0),
                        'moment_end_frame': annotation.get('moment_end_frame', 0),
                        'verb': annotation.get('verb', ''),
                        'noun': annotation.get('noun', ''),
                        'verb_noun': annotation.get('verb_noun', ''),
                        'verb_noun_uid': annotation.get('verb_noun_uid', ''),
                        'verb_uid': annotation.get('verb_uid', ''),
                        'noun_uid': annotation.get('noun_uid', ''),
                        'video_start_sec': annotation.get('video_start_sec', 0),
                        'video_end_sec': annotation.get('video_end_sec', 0),
                        'video_start_frame': annotation.get('video_start_frame', 0),
                        'video_end_frame': annotation.get('video_end_frame', 0)
                    }
                    moments_annotations.append(moment_info)
                break
    
    return moments_annotations

def process_video_moments(video_info: Dict, moments_data: Dict) -> Dict:
    """Process moments annotations for a single video."""
    video_uid = video_info['video_uid']
    clip_uid = video_info['clip_uid']
    original_summary = video_info['original_summary']
    
    print(f"\nProcessing video: {video_uid} (clip: {clip_uid})")
    
    # Find video file path
    video_path = find_video_path(clip_uid)
    
    # Initialize result structure
    result = {
        'video_uid': video_uid,
        'clip_uid': clip_uid,
        'video_path': video_path,
        'original_summary': original_summary,
        'moments_annotations': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract moments annotations
    moments_annotations = extract_moments_annotations(moments_data, clip_uid)
    result['moments_annotations'] = moments_annotations
    result['total_moments'] = len(moments_annotations)
    
    print(f"Found {len(moments_annotations)} moments annotations")
    
    # Print some examples
    if moments_annotations:
        print("Example moments:")
        for i, moment in enumerate(moments_annotations[:3]):  # Show first 3
            print(f"  {i+1}. {moment.get('verb', '')} {moment.get('noun', '')} "
                  f"({moment.get('moment_start_sec', 0):.1f}s - {moment.get('moment_end_sec', 0):.1f}s)")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract Ego4D moments annotations for NLQ validation dataset')
    parser.add_argument('--moments_path', type=str, 
                       default='../remote_ego4d/v2/annotations/moments_val.json',
                       help='Path to moments annotations JSON file')
    parser.add_argument('--output_file', type=str, 
                       default='ego4d_nlq_moments_annotations.json',
                       help='Output JSON file path')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    print("Ego4D Moments Annotations Extractor for NLQ Validation Dataset")
    print("=" * 70)
    print(f"Moments path: {args.moments_path}")
    print(f"Output file: {args.output_file}")
    print(f"Max videos: {args.max_videos or 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load video data
    video_data = load_nlq_val_videos()
    if args.max_videos:
        video_data = video_data[:args.max_videos]
        print(f"Processing first {len(video_data)} videos")
    
    # Load moments annotation data
    moments_data = None
    if os.path.exists(args.moments_path):
        try:
            moments_data = load_ego4d_annotations(args.moments_path)
            print(f"Loaded moments data with {len(moments_data.get('videos', []))} videos")
        except Exception as e:
            print(f"Error loading moments data: {e}")
            return
    else:
        print(f"Moments file not found: {args.moments_path}")
        return
    
    # Process videos
    all_results = []
    videos_with_moments = 0
    total_moments = 0
    
    for i, video_info in enumerate(tqdm(video_data, desc="Processing videos")):
        if not video_info['has_clips']:
            print(f"Video {video_info['video_uid']} has no clips, skipping...")
            continue
        
        try:
            result = process_video_moments(video_info, moments_data)
            
            if result:
                all_results.append(result)
                
                if result['total_moments'] > 0:
                    videos_with_moments += 1
                    total_moments += result['total_moments']
                
                # Save incrementally
                with open(args.output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"Completed video {video_info['video_uid']}, saved to {args.output_file}")
            
        except Exception as e:
            print(f"Error processing video {video_info['video_uid']}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(all_results)}")
    print(f"Videos with moments annotations: {videos_with_moments}")
    print(f"Total moments annotations: {total_moments}")
    print(f"Average moments per video (with moments): {total_moments/videos_with_moments if videos_with_moments > 0 else 0:.2f}")
    print(f"Results saved to: {args.output_file}")
    
    # Print summary statistics
    if all_results:
        # Count unique verbs and nouns
        all_verbs = set()
        all_nouns = set()
        for result in all_results:
            for moment in result['moments_annotations']:
                if moment.get('verb'):
                    all_verbs.add(moment['verb'])
                if moment.get('noun'):
                    all_nouns.add(moment['noun'])
        
        print(f"Unique verbs found: {len(all_verbs)}")
        print(f"Unique nouns found: {len(all_nouns)}")
        
        # Show most common verbs and nouns
        verb_counts = {}
        noun_counts = {}
        for result in all_results:
            for moment in result['moments_annotations']:
                verb = moment.get('verb', '')
                noun = moment.get('noun', '')
                if verb:
                    verb_counts[verb] = verb_counts.get(verb, 0) + 1
                if noun:
                    noun_counts[noun] = noun_counts.get(noun, 0) + 1
        
        print("\nTop 10 most common verbs:")
        for verb, count in sorted(verb_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {verb}: {count}")
        
        print("\nTop 10 most common nouns:")
        for noun, count in sorted(noun_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {noun}: {count}")

if __name__ == "__main__":
    main() 