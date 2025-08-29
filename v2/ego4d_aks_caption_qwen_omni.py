#!/usr/bin/env python3
"""
Qwen Omni Ego4D AKS Frame Captioning and Summarization (Qwen Omni GPU Optimized)
Processes ego4d_aks_results.json format and generates captions for selected frames.
Uses Qwen Omni for proper GPU inference with the Qwen Omni model.
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
    """Load Qwen Omni model for GPU inference."""
    print("Loading Qwen Omni model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        # Replace this with the actual Qwen Omni pipeline/model loading code
        from qwen_omni import pipeline
        
        # Create pipeline with Qwen Omni model
        pipe = pipeline(
            'Qwen/Qwen-Omni',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Qwen Omni model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"Error loading Qwen Omni model: {e}")
        traceback.print_exc()
        return None

def extract_frames_from_directory(video_uid, frame_indices, frames_dir="../v1/ego4d_aks_full/frames"):
    """Extract frames from the frames directory using frame indices."""
    frames = []
    video_frames_dir = os.path.join(frames_dir, video_uid)
    
    if not os.path.exists(video_frames_dir):
        print(f"Frames directory not found: {video_frames_dir}")
        return frames
    
    # Get all frame files in the directory
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print(f"No frame files found in: {video_frames_dir}")
        return frames
    
    print(f"Found {len(frame_files)} frame files")
    print(f"Frame indices to extract: {frame_indices[:5]}...")  # Show first 5 indices
    
    # The frame files are already in the correct order, so we can use the frame_indices directly
    # as indices into the frame_files list
    for i, frame_idx in enumerate(frame_indices):
        if i < len(frame_files):  # Use sequential index, not frame_idx
            frame_path = os.path.join(video_frames_dir, frame_files[i])
            try:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    print(f"Loaded frame {i}: {frame_files[i]}")
                else:
                    print(f"Failed to load frame {i}: {frame_path}")
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
    
    return frames

def caption_frame(pipe, image):
    """Caption a single frame using Qwen Omni pipeline."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Use Qwen Omni pipeline for inference
        prompt = "Describe this image briefly."
        response = pipe([(prompt, [image])])
        
        # Debug: print the response
        print(f"Generated caption: {response}")
        
        # Convert response to string if it's a Response object
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
    """Create concise summary from frame captions using Qwen Omni."""
    if not captions:
        return "No captions available."
    
    try:
        # Join captions
        context = " ".join(captions)
        
        # Use Qwen Omni pipeline for text-only generation (no image)
        prompt = f"Summarize the main activity in these frame descriptions in one short sentence: {context}"
        response = pipe([prompt])
        
        # Convert response to string if it's a Response object
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

def clear_output_file(output_file):
    """Clear the output file to remove already summarized videos."""
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared existing output file: {output_file}")

def load_processed_video_uids(output_file):
    """Load already processed video_uids from the output JSON file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                data = json.load(f)
                for entry in data:
                    if entry.get('generated_summary'):
                        processed.add(entry['video_uid'])
            except:
                pass
    return processed

def process_ego4d_aks_videos(aks_file, pipe, output_file, max_videos=None):
    """Process videos using ego4d_aks_results.json format, saving results to JSON file with frame-level resumption."""
    # Load AKS results
    with open(aks_file, 'r') as f:
        aks_results = json.load(f)
    print(f"Loaded {len(aks_results)} videos from AKS results")
    
    if max_videos:
        aks_results = aks_results[:max_videos]
        print(f"Processing first {len(aks_results)} videos")
    
    # Load existing results if file exists
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []
    
    # Create a map of existing results for quick lookup
    existing_results_map = {result['video_uid']: result for result in existing_results}
    
    processed_count = 0
    all_results = existing_results.copy()
    
    for video_data in tqdm(aks_results, desc="Processing videos"):
        video_uid = video_data['video_uid']
        original_video_path = video_data.get('video_path', '')
        original_summary = video_data.get('summary', '')
        selected_frames = video_data.get('selected_frames', [])
        
        print(f"\nProcessing: {video_uid}")
        print(f"Original path: {original_video_path}")
        print(f"Selected frames: {len(selected_frames)}")
        
        if not selected_frames:
            print(f"No selected frames for video: {video_uid}")
            continue
            
        try:
            # Check if video already exists and has complete frame captions
            existing_result = existing_results_map.get(video_uid)
            if existing_result and existing_result.get('generated_summary'):
                print(f"Video {video_uid} already has complete results, skipping...")
                continue
            
            # Extract frames from directory
            frames = extract_frames_from_directory(video_uid, selected_frames)
            print(f"Extracted {len(frames)} frames")
            
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            
            # Generate captions for all frames
            frame_captions = []
            for i, frame in enumerate(tqdm(frames, desc="Captioning frames")):
                caption = caption_frame(pipe, frame)
                frame_captions.append(caption)
                print(f"Frame {i}: {caption[:100]}...")  # Debug output
                
            # Generate summary for all captions
            video_summary = summarize_captions(pipe, frame_captions)
            
            # Create the final result
            result = {
                'video_uid': video_uid,
                'original_video_path': original_video_path,
                'original_summary': original_summary,
                'selected_frames': selected_frames,
                'frame_captions': frame_captions,
                'generated_summary': video_summary
            }
            
            # Update or add result
            if video_uid in existing_results_map:
                # Update existing result
                for j, existing_result in enumerate(all_results):
                    if existing_result['video_uid'] == video_uid:
                        all_results[j] = result
                        break
            else:
                # Add new result
                all_results.append(result)
            
            # Save final result
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            processed_count += 1
            print(f"Completed and saved result for {video_uid}")
            
        except Exception as e:
            print(f"Error processing video {video_uid}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully processed {processed_count} new videos. Total results: {len(all_results)}")
    print(f"Results saved to {output_file}")

def main():
    """Main function to run the Ego4D AKS captioning pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ego4d_captioning.log'),
            logging.StreamHandler()
        ]
    )
    
    # Configuration
    aks_file = "../v1/ego4d_aks_full/ego4d_aks_results.json"
    output_file = "ego4d_aks_captions.json"  # Use the existing results file
    max_videos = None  # Set to a number to limit processing
    
    print("Qwen Omni Ego4D AKS Captioning")
    print("=" * 50)
    print(f"AKS file: {aks_file}")
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Check if AKS file exists
    if not os.path.exists(aks_file):
        print(f"Error: AKS file not found: {aks_file}")
        return
    
    # Load model
    pipe = load_model()
    if pipe is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process videos
    process_ego4d_aks_videos(aks_file, pipe, output_file, max_videos)
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main() 