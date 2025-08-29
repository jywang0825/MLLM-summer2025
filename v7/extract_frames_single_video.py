#!/usr/bin/env python3
"""
Extract 32 uniform frames from the first video in the NLQ dataset
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import traceback
import sys
from datetime import datetime

def find_video_path(clip_uid, clips_manifest_path="../remote_ego4d/v2/clips/manifest.csv"):
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

def extract_uniform_frames_from_video(video_path, num_frames=32, output_dir="extracted_frames"):
    """Extract num_frames uniformly from a video file and save them as images."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
        
        frames = []
        if total_frames <= num_frames:
            # If video has fewer frames than requested, take all frames
            frame_indices = list(range(total_frames))
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        print(f"Extracting {len(frame_indices)} frames at indices: {frame_indices}")
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Save frame as image
                frame_pil = Image.fromarray(frame_rgb)
                frame_filename = f"frame_{i:03d}_idx_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                frame_pil.save(frame_path, quality=95)
                
                print(f"Saved frame {i+1}/{len(frame_indices)}: {frame_filename} (original frame {frame_idx})")
            else:
                print(f"Failed to read frame {frame_idx}")
        
        cap.release()
        print(f"Successfully extracted and saved {len(frames)} frames to {output_dir}")
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        traceback.print_exc()
        return []

def load_nlq_val_videos(summaries_path="../v1/nlq_val_summaries.json", nlq_val_path="../remote_ego4d/v2/annotations/nlq_val.json"):
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

def main():
    print("Extract 32 uniform frames from first NLQ video")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Load video data
    video_data = load_nlq_val_videos()
    if not video_data:
        print("No video data loaded. Exiting.")
        return
    
    # Find first video with clips
    first_video = None
    for video_info in video_data:
        if video_info['has_clips']:
            first_video = video_info
            break
    
    if not first_video:
        print("No videos with clips found. Exiting.")
        return
    
    video_uid = first_video['video_uid']
    clip_uid = first_video['clip_uid']
    original_summary = first_video['original_summary']
    
    print(f"\nProcessing first video:")
    print(f"Video UID: {video_uid}")
    print(f"Clip UID: {clip_uid}")
    print(f"Original summary: {original_summary}")
    
    # Find video file path using clip_uid
    video_path = find_video_path(clip_uid)
    if not video_path:
        print(f"Could not find video path for clip {clip_uid}. Exiting.")
        return
    
    print(f"Video path: {video_path}")
    
    # Create output directory with video UID
    output_dir = f"extracted_frames_{video_uid}"
    
    # Extract frames
    frames = extract_uniform_frames_from_video(video_path, num_frames=32, output_dir=output_dir)
    
    if frames:
        print(f"\nSuccessfully extracted {len(frames)} frames!")
        print(f"Frames saved to: {output_dir}")
        print(f"Frame dimensions: {frames[0].shape}")
    else:
        print("Failed to extract frames.")

if __name__ == "__main__":
    main() 