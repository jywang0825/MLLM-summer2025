#!/usr/bin/env python3
"""
Ego4D Uniform Frame Captioning using a simpler image captioning model
Samples frames uniformly from each video and generates captions for those frames.
Uses a standard image captioning model to avoid FlashAttention2 CUDA issues.
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
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM

warnings.filterwarnings("ignore")

def load_model():
    """Load a simple image captioning model."""
    print("Loading image captioning model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        # Use a simpler model that doesn't have FlashAttention2 issues
        model_name = "microsoft/git-base-coco"  # Simple image captioning model
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"Model {model_name} loaded successfully!")
        return processor, model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None, None

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

def caption_frames_simple(processor, model, frames, video_time=0.0):
    """Caption frames using a simple image captioning model."""
    try:
        if not frames:
            return ["No frames available"]
        
        captions = []
        for i, frame in enumerate(frames):
            try:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Process image
                inputs = processor(images=pil_image, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate caption
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        num_beams=1
                    )
                
                # Decode caption
                caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                captions.append(caption)
                print(f"Frame {i}: {caption[:100]}...")
                
            except Exception as e:
                print(f"Error captioning frame {i}: {e}")
                captions.append(f"Error: {str(e)}")
        
        return captions
        
    except Exception as e:
        print(f"Error in caption_frames_simple: {e}")
        traceback.print_exc()
        return [f"Error: {str(e)}"]

def summarize_captions_simple(captions):
    """Create concise summary from frame captions."""
    if not captions:
        return "No captions available."
    
    try:
        # Simple summarization: take the most common words/phrases
        context = " ".join(captions)
        
        # Extract key activities (simple approach)
        words = context.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get most common words
        from collections import Counter
        word_counts = Counter(filtered_words)
        common_words = [word for word, count in word_counts.most_common(10)]
        
        summary = f"Main activities: {', '.join(common_words[:5])}"
        return summary
        
    except Exception as e:
        print(f"Error in summarize_captions_simple: {e}")
        traceback.print_exc()
        return f"Summary error: {str(e)}"

def process_nlq_videos_uniform(processor, model, output_file, max_videos=None, num_frames=32):
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
            
            # Generate captions for frames
            frame_captions = caption_frames_simple(processor, model, frames)
            
            # Create video summary
            video_summary = summarize_captions_simple(frame_captions)
            
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ego4d_captioning_simple.log'),
            logging.StreamHandler()
        ]
    )
    
    output_file = "ego4d_uniform_captions_simple.json"
    max_videos = 3  # Test with 3 videos first
    num_frames = 16  # Use fewer frames for testing
    
    print("Ego4D Uniform Frame Captioning (Simple Model)")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All'}")
    print(f"Frames per video: {num_frames}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    processor, model = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process videos
    results = process_nlq_videos_uniform(processor, model, output_file, max_videos, num_frames)
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 