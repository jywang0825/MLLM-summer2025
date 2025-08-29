#!/usr/bin/env python3
"""
InternVL3 Ego4D NLQ Validation Uniform Frame Captioning and Summarization (LMDeploy GPU Optimized)
Extracts frames directly from video files and samples 32 frames uniformly from each video.
Uses LMDeploy for proper GPU inference with InternVL3 model.
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import traceback
import sys
import torch
import time
from datetime import datetime
from torchvision import transforms
import logging

def load_model():
    """Load InternVL3 model using LMDeploy for GPU inference."""
    print("Loading InternVL3 model with LMDeploy...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        from lmdeploy import pipeline
        
        # Create pipeline with InternVL3 model
        pipe = pipeline(
            'OpenGVLab/InternVL3-8B',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("InternVL3 model loaded successfully with LMDeploy!")
        return pipe
        
    except Exception as e:
        print(f"Error loading InternVL model with LMDeploy: {e}")
        traceback.print_exc()
        return None

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

def extract_uniform_frames_from_video(video_path, num_frames=32):
    """Extract num_frames uniformly from a video file."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []
    
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
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                print(f"Extracted frame {i+1}/{len(frame_indices)}: frame {frame_idx}")
            else:
                print(f"Failed to read frame {frame_idx}")
        
        cap.release()
        print(f"Successfully extracted {len(frames)} frames from {video_path}")
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        traceback.print_exc()
        return []

def caption_frame(pipe, image):
    """Caption a single frame using LMDeploy InternVL pipeline."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        prompt = "Describe this image briefly."
        response = pipe([(prompt, [image])])
        print(f"Generated caption: {response}")
        if hasattr(response, '__iter__') and len(response) > 0:
            caption = response[0]
            if hasattr(caption, 'text'):
                return caption.text
            else:
                return str(caption)
        else:
            return "Error: No response generated"
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def summarize_captions(pipe, captions):
    """Create concise summary from frame captions using LMDeploy."""
    if not captions:
        return "No captions available."
    try:
        context = " ".join(captions)
        prompt = f"Summarize the main activity in these frame descriptions in one short sentence: {context}"
        response = pipe([prompt])
        if hasattr(response, '__iter__') and len(response) > 0:
            summary = response[0]
            if hasattr(summary, 'text'):
                return summary.text
            else:
                return str(summary)
        else:
            return "Error: No response generated"
    except Exception as e:
        print(f"Error in summarize_captions: {e}")
        traceback.print_exc()
        return f"Summary error: {str(e)}"

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

def process_nlq_videos_uniform(pipe, output_file, max_videos=None, num_frames=32):
    """Process NLQ validation videos with uniform frame sampling."""
    video_data = load_nlq_val_videos()
    if max_videos:
        video_data = video_data[:max_videos]
        print(f"Processing first {len(video_data)} videos")
    
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []
    
    existing_uids = {result['video_uid'] for result in existing_results}
    all_results = existing_results.copy()
    
    for i, video_info in enumerate(tqdm(video_data, desc="Processing videos")):
        video_uid = video_info['video_uid']
        clip_uid = video_info['clip_uid']
        original_summary = video_info['original_summary']
        has_clips = video_info['has_clips']
        
        if video_uid in existing_uids:
            print(f"Video {video_uid} already processed, skipping...")
            continue
        
        if not has_clips:
            print(f"Video {video_uid} has no clips, skipping...")
            continue
            
        print(f"\nProcessing video {i+1}/{len(video_data)}: {video_uid} (clip: {clip_uid})")
        print(f"Original summary: {original_summary}")
        
        # Find video file path using clip_uid
        video_path = find_video_path(clip_uid)
        if not video_path:
            print(f"Could not find video path for clip {clip_uid}, skipping...")
            continue
        
        try:
            # Extract frames uniformly
            frames = extract_uniform_frames_from_video(video_path, num_frames)
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            
            # Generate captions for each frame
            frame_captions = []
            for j, frame in enumerate(tqdm(frames, desc="Captioning frames")):
                caption = caption_frame(pipe, frame)
                frame_captions.append(caption)
                print(f"Frame {j}: {caption[:100]}...")
            
            # Create video summary
            video_summary = summarize_captions(pipe, frame_captions)
            
            # Create result entry
            result = {
                'video_uid': video_uid,
                'video_path': video_path,
                'original_summary': original_summary,
                'uniform_sampled_frames': len(frames),
                'frame_captions': frame_captions,
                'generated_summary': video_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Completed video {video_uid}, saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing video {video_uid}: {e}")
            traceback.print_exc()
            continue
    
    return all_results

def main():
    print("InternVL3 Ego4D NLQ Validation Uniform Frame Captioning with LMDeploy")
    print("=" * 80)
    print(f"Output file: ego4d_nlq_uniform_captions.json")
    print(f"Max videos: All")
    print(f"Frames per video: 32")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load model
    pipe = load_model()
    if not pipe:
        print("Failed to load model. Exiting.")
        return
    
    # Process videos
    output_file = "ego4d_nlq_uniform_captions.json"
    results = process_nlq_videos_uniform(pipe, output_file)
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 