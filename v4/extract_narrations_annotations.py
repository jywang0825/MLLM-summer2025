#!/usr/bin/env python3
"""
Ego4D Narrations Annotations Extractor for NLQ Validation Dataset (Uniform Frames)
Extracts narrations for 32 uniformly sampled frames per video in the NLQ validation set.
Simple and reliable version.
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
import ijson  # For parsing large JSON files efficiently
from decimal import Decimal
import numpy as np
import cv2
from pathlib import Path

def load_existing_results(output_file: str) -> Dict[str, Dict]:
    """Load existing results to avoid re-processing videos."""
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                for video_data in existing_data:
                    video_uid = video_data.get('video_uid')
                    if video_uid:
                        existing_results[video_uid] = video_data
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    return existing_results

def load_nlq_val_videos(summaries_path: str = "../v1/nlq_val_summaries.json", 
                       nlq_val_path: str = "../remote_ego4d/v2/annotations/nlq_val.json") -> List[Dict]:
    try:
        with open(summaries_path, 'r') as f:
            summaries_data = json.load(f)
        with open(nlq_val_path, 'r') as f:
            nlq_data = json.load(f)
        video_clips_map = {}
        for video in nlq_data['videos']:
            if video['clips']:
                video_clips_map[video['video_uid']] = video['clips'][0]['clip_uid']
        video_data = []
        for summary_item in summaries_data:
            video_uid = summary_item['video_uid']
            original_summary = summary_item['summary']
            if video_uid in video_clips_map:
                video_data.append({
                    'video_uid': video_uid,
                    'clip_uid': video_clips_map[video_uid],
                    'original_summary': original_summary,
                    'has_clips': True
                })
            else:
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
    try:
        df = pd.read_csv(clips_manifest_path)
        match = df[df['exported_clip_uid'] == clip_uid]
        if not match.empty:
            manifold_path = match.iloc[0]['manifold_location']
            local_path = manifold_path.replace('manifold://ego4d_fair/tree/exported_clips/', '../remote_ego4d/v2/clips/')
            return local_path
        else:
            print(f"Clip UID {clip_uid} not found in manifest")
            return None
    except Exception as e:
        print(f"Error finding video path for {clip_uid}: {e}")
        return None

def extract_uniform_frame_timestamps(video_path: str, num_frames: int = 32) -> Tuple[list, list]:
    """Extract uniform frame timestamps using OpenCV."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return [], []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return [], []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if fps <= 0:
            print(f"Invalid FPS for video: {video_path}")
            return [], []
        
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        timestamps = [i / fps for i in frame_indices]
        print(f"Extracted {len(frame_indices)} uniform frames, {fps:.2f} FPS, {total_frames} total frames")
        return frame_indices, timestamps
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        traceback.print_exc()
        return [], []

def extract_narrations_for_video(narrations_file: str, video_uid: str) -> List[Dict]:
    """Extract narrations for a specific video using ijson."""
    narrations = []
    
    try:
        with open(narrations_file, 'rb') as f:
            parser = ijson.parse(f)
            current_video = None
            in_video = False
            in_narrations = False
            
            for prefix, event, value in parser:
                if prefix == f'videos.{video_uid}' and event == 'start_map':
                    in_video = True
                    current_video = video_uid
                elif in_video and prefix == f'videos.{video_uid}.narrations' and event == 'start_array':
                    in_narrations = True
                elif in_narrations and prefix.startswith(f'videos.{video_uid}.narrations.item'):
                    if event == 'start_map':
                        narration = {}
                    elif event == 'end_map':
                        narrations.append(narration)
                    elif event in ('string', 'number'):
                        field = prefix.split('.')[-1]
                        narration[field] = value
                elif in_video and prefix == f'videos.{video_uid}.narrations' and event == 'end_array':
                    in_narrations = False
                elif in_video and prefix == f'videos.{video_uid}' and event == 'end_map':
                    in_video = False
                    break
                        
    except Exception as e:
        print(f"Error extracting narrations for video {video_uid}: {e}")
        traceback.print_exc()
    
    return narrations

def find_closest_narration(frame_time: float, narrations: List[Dict], window: float = 0.5) -> List[Dict]:
    """Find narrations within ±window seconds of the frame_time, or the closest if none in window."""
    if not narrations:
        return []
    # Convert all narration times to float
    narration_times = [float(n.get('time', 0)) for n in narrations]
    # Find narrations within window
    close_narrs = [n for n, t in zip(narrations, narration_times) if abs(t - frame_time) <= window]
    if close_narrs:
        return close_narrs
    # If none in window, return the closest one
    closest_idx = int(np.argmin([abs(t - frame_time) for t in narration_times]))
    return [narrations[closest_idx]]

def process_video_narrations_uniform(video_info: Dict, narrations_file: str, num_frames: int = 32) -> Optional[Dict]:
    video_uid = video_info['video_uid']
    clip_uid = video_info['clip_uid']
    original_summary = video_info['original_summary']
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_uid}")
    print(f"Clip UID: {clip_uid}")
    print(f"Summary: {original_summary[:100]}{'...' if len(original_summary) > 100 else ''}")
    print(f"{'='*60}")
    
    video_path = find_video_path(clip_uid)
    if not video_path:
        print(f"❌ Could not find video path for clip {clip_uid}")
        return {
            'video_uid': video_uid,
            'clip_uid': clip_uid,
            'video_path': None,
            'original_summary': original_summary,
            'error': 'Video path not found',
            'uniform_sampled_frames': 0,
            'frame_indices': [],
            'frame_timestamps': [],
            'frame_narrations': [],
            'frame_narration_map': {},
            'total_narrations': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    print(f"📁 Video path: {video_path}")
    
    frame_indices, frame_timestamps = extract_uniform_frame_timestamps(video_path, num_frames)
    if not frame_indices:
        print(f"❌ Failed to extract frame timestamps")
        return None
    
    print(f"✅ Extracted {len(frame_indices)} uniform frames")
    print(f"📊 Video duration: {frame_timestamps[-1]:.2f} seconds")
    
    result = {
        'video_uid': video_uid,
        'clip_uid': clip_uid,
        'video_path': video_path,
        'original_summary': original_summary,
        'uniform_sampled_frames': len(frame_indices),
        'frame_indices': frame_indices,
        'frame_timestamps': frame_timestamps,
        'frame_narrations': [],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"🔍 Extracting narrations...")
    narrations = extract_narrations_for_video(narrations_file, video_uid)
    result['total_narrations'] = len(narrations)
    print(f"✅ Found {len(narrations)} narrations")
    
    frame_narr_map = {}
    print(f"🔗 Matching narrations to frames...")
    for i, ts in enumerate(frame_timestamps):
        narrs = find_closest_narration(ts, narrations, window=0.5)
        frame_narr_map[i] = narrs
        result['frame_narrations'].append({
            'frame_index': i,
            'timestamp': ts,
            'narrations': narrs
        })
    
    result['frame_narration_map'] = frame_narr_map
    
    # Count frames with narrations
    frames_with_narrations = sum(1 for narrs in frame_narr_map.values() if narrs)
    print(f"✅ {frames_with_narrations}/{len(frame_indices)} frames have narrations")
    print(f"✅ Video processing completed successfully!")
    
    return result

def save_results(all_results: List[Dict], output_file: str):
    """Save results to file with proper formatting."""
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract Ego4D narrations for uniformly sampled frames in NLQ validation dataset')
    parser.add_argument('--narrations_path', type=str, 
                       default='../remote_ego4d/v2/annotations/all_narrations_redacted.json',
                       help='Path to all narrations redacted JSON file')
    parser.add_argument('--output_file', type=str, 
                       default='ego4d_nlq_uniform_frame_narrations.json',
                       help='Output JSON file path')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample uniformly')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process')
    parser.add_argument('--force', action='store_true',
                       help='Force re-processing of already processed videos')
    args = parser.parse_args()
    
    print("🎬 Ego4D Narrations Annotations Extractor for NLQ Validation Dataset (Uniform Frames)")
    print("=" * 80)
    print(f"📂 Narrations path: {args.narrations_path}")
    print(f"💾 Output file: {args.output_file}")
    print(f"🎞️  Frames per video: {args.num_frames}")
    print(f"📊 Max videos: {args.max_videos or 'All'}")
    print(f"🔄 Force re-processing: {args.force}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not os.path.exists(args.narrations_path):
        print(f"❌ Narrations file not found: {args.narrations_path}")
        return
    
    # Load existing results
    existing_results = load_existing_results(args.output_file)
    
    # Load video data
    video_data = load_nlq_val_videos()
    if args.max_videos:
        video_data = video_data[:args.max_videos]
        print(f"📋 Processing first {len(video_data)} videos")
    
    # Filter out already processed videos (unless force is enabled)
    if not args.force and existing_results:
        original_count = len(video_data)
        video_data = [v for v in video_data if v['video_uid'] not in existing_results]
        skipped_count = original_count - len(video_data)
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} already processed videos")
    
    if not video_data:
        print("✅ All videos have already been processed!")
        return
    
    print(f"🚀 Starting processing of {len(video_data)} videos...")
    
    all_results = list(existing_results.values())  # Start with existing results
    processed_count = 0
    error_count = 0
    
    # Process videos one by one
    for i, video_info in enumerate(video_data):
        print(f"\n📹 Processing video {i+1}/{len(video_data)}")
        
        result = process_video_narrations_uniform(video_info, args.narrations_path, args.num_frames)
        
        if result:
            all_results.append(result)
            processed_count += 1
        else:
            error_count += 1
        
        # Save after each video (incremental saving)
        save_results(all_results, args.output_file)
        
        print(f"✅ Video {i+1} completed: {processed_count} processed, {error_count} errors")
    
    print(f"\n🎉 Processing completed!")
    print(f"📊 Summary:")
    print(f"   ✅ Total videos processed: {processed_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   💾 Total results in file: {len(all_results)}")
    print(f"   📁 Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 