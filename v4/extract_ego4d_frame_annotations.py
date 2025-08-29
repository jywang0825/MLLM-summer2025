#!/usr/bin/env python3
"""
Ego4D Frame Annotations Extractor for Uniformly Selected Frames
Extracts frame-level annotations from Ego4D dataset for uniformly sampled frames.
Supports moments, narrations, and other annotation types.
"""

import os
import json
import cv2
import numpy as np
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

def extract_uniform_frames_from_video(video_path: str, num_frames: int = 32) -> Tuple[List[int], List[float]]:
    """Extract frame indices and timestamps for uniform sampling."""
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
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
        
        frame_indices = []
        timestamps = []
        
        if total_frames <= num_frames:
            # If video has fewer frames than requested, take all frames
            frame_indices = list(range(total_frames))
            timestamps = [i / fps for i in frame_indices]
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            timestamps = [i / fps for i in frame_indices]
        
        cap.release()
        print(f"Selected {len(frame_indices)} frames uniformly from {video_path}")
        return frame_indices, timestamps
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        traceback.print_exc()
        return [], []

def extract_moments_annotations(moments_data: Dict, clip_uid: str, frame_timestamps: List[float]) -> List[Dict]:
    """Extract moments annotations for the given clip and frame timestamps."""
    frame_annotations = []
    
    # Find the clip in moments data
    for video in moments_data.get('videos', []):
        for clip in video.get('clips', []):
            if clip.get('clip_uid') == clip_uid:
                # Get annotations for this clip
                for annotation in clip.get('annotations', []):
                    for label in annotation.get('labels', []):
                        start_time = label.get('start_time', 0)
                        end_time = label.get('end_time', 0)
                        label_text = label.get('label', '')
                        primary = label.get('primary', False)
                        
                        # Find frames that fall within this annotation
                        for i, timestamp in enumerate(frame_timestamps):
                            if start_time <= timestamp <= end_time:
                                frame_annotations.append({
                                    'frame_index': i,
                                    'timestamp': timestamp,
                                    'annotation_type': 'moments',
                                    'label': label_text,
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'primary': primary,
                                    'annotator_uid': annotation.get('annotator_uid', '')
                                })
                break
    
    return frame_annotations

def extract_narration_annotations(narration_data: Dict, video_uid: str, frame_timestamps: List[float]) -> List[Dict]:
    """Extract narration annotations for the given video and frame timestamps."""
    frame_annotations = []
    
    # Find the video in narration data
    video_info = narration_data.get('videos', {}).get(video_uid)
    if not video_info:
        return frame_annotations
    
    narrations = video_info.get('narrations', [])
    
    for narration in narrations:
        narration_time = narration.get('time', 0)
        narration_text = narration.get('text', '')
        
        # Find the closest frame to this narration
        if frame_timestamps:
            closest_frame_idx = min(range(len(frame_timestamps)), 
                                  key=lambda i: abs(frame_timestamps[i] - narration_time))
            closest_timestamp = frame_timestamps[closest_frame_idx]
            
            # Only include if within 2 seconds of the frame
            if abs(closest_timestamp - narration_time) <= 2.0:
                frame_annotations.append({
                    'frame_index': closest_frame_idx,
                    'timestamp': closest_timestamp,
                    'annotation_type': 'narration',
                    'narration_time': narration_time,
                    'narration_text': narration_text,
                    'time_diff': abs(closest_timestamp - narration_time)
                })
    
    return frame_annotations

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

def process_video_annotations(video_info: Dict, moments_data: Optional[Dict] = None, 
                            narration_data: Optional[Dict] = None, num_frames: int = 32) -> Dict:
    """Process annotations for a single video."""
    video_uid = video_info['video_uid']
    clip_uid = video_info['clip_uid']
    original_summary = video_info['original_summary']
    
    print(f"\nProcessing video: {video_uid} (clip: {clip_uid})")
    
    # Find video file path
    video_path = find_video_path(clip_uid)
    if not video_path:
        print(f"Could not find video path for clip {clip_uid}")
        return None
    
    # Extract uniform frame indices and timestamps
    frame_indices, frame_timestamps = extract_uniform_frames_from_video(video_path, num_frames)
    if not frame_indices:
        print(f"No frames extracted for video: {video_uid}")
        return None
    
    # Initialize result structure
    result = {
        'video_uid': video_uid,
        'clip_uid': clip_uid,
        'video_path': video_path,
        'original_summary': original_summary,
        'uniform_sampled_frames': len(frame_indices),
        'frame_indices': frame_indices,
        'frame_timestamps': frame_timestamps,
        'frame_annotations': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Extract moments annotations
    if moments_data:
        moments_annotations = extract_moments_annotations(moments_data, clip_uid, frame_timestamps)
        result['frame_annotations'].extend(moments_annotations)
        print(f"Found {len(moments_annotations)} moments annotations")
    
    # Extract narration annotations
    if narration_data:
        narration_annotations = extract_narration_annotations(narration_data, video_uid, frame_timestamps)
        result['frame_annotations'].extend(narration_annotations)
        print(f"Found {len(narration_annotations)} narration annotations")
    
    # Group annotations by frame
    frame_annotation_map = {}
    for annotation in result['frame_annotations']:
        frame_idx = annotation['frame_index']
        if frame_idx not in frame_annotation_map:
            frame_annotation_map[frame_idx] = []
        frame_annotation_map[frame_idx].append(annotation)
    
    result['frame_annotation_map'] = frame_annotation_map
    result['total_annotations'] = len(result['frame_annotations'])
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract Ego4D frame annotations for uniformly selected frames')
    parser.add_argument('--moments_path', type=str, 
                       default='../remote_ego4d/v2/annotations/moments_val.json',
                       help='Path to moments annotations JSON file')
    parser.add_argument('--narration_path', type=str, 
                       default='../remote_ego4d/v2/annotations/narration.json',
                       help='Path to narration annotations JSON file')
    parser.add_argument('--output_file', type=str, 
                       default='ego4d_uniform_frame_annotations.json',
                       help='Output JSON file path')
    parser.add_argument('--num_frames', type=int, default=32,
                       help='Number of frames to sample uniformly')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process')
    
    args = parser.parse_args()
    
    print("Ego4D Uniform Frame Annotations Extractor")
    print("=" * 50)
    print(f"Moments path: {args.moments_path}")
    print(f"Narration path: {args.narration_path}")
    print(f"Output file: {args.output_file}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Max videos: {args.max_videos or 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Load video data
    video_data = load_nlq_val_videos()
    if args.max_videos:
        video_data = video_data[:args.max_videos]
        print(f"Processing first {len(video_data)} videos")
    
    # Load annotation data
    moments_data = None
    if os.path.exists(args.moments_path):
        try:
            moments_data = load_ego4d_annotations(args.moments_path)
            print(f"Loaded moments data with {len(moments_data.get('videos', []))} videos")
        except Exception as e:
            print(f"Error loading moments data: {e}")
    
    narration_data = None
    if os.path.exists(args.narration_path):
        try:
            narration_data = load_ego4d_annotations(args.narration_path)
            print(f"Loaded narration data with {len(narration_data.get('videos', {}))} videos")
        except Exception as e:
            print(f"Error loading narration data: {e}")
    
    # Process videos
    all_results = []
    
    for i, video_info in enumerate(tqdm(video_data, desc="Processing videos")):
        if not video_info['has_clips']:
            print(f"Video {video_info['video_uid']} has no clips, skipping...")
            continue
        
        try:
            result = process_video_annotations(
                video_info, 
                moments_data, 
                narration_data, 
                args.num_frames
            )
            
            if result:
                all_results.append(result)
                
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
    print(f"Results saved to: {args.output_file}")
    
    # Print summary statistics
    if all_results:
        total_annotations = sum(r['total_annotations'] for r in all_results)
        avg_annotations_per_video = total_annotations / len(all_results)
        print(f"Total annotations: {total_annotations}")
        print(f"Average annotations per video: {avg_annotations_per_video:.2f}")

if __name__ == "__main__":
    main() 