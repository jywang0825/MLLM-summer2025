#!/usr/bin/env python3
"""
InternVL3 Ego4D AKS Frame Captioning and Summarization
Processes ego4d_aks_results.json format and generates captions for selected frames.
Uses the official InternVL repository for proper multimodal inference.
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
from torchvision import transforms

# Add InternVL to path
sys.path.append('../InternVL/internvl_chat')
sys.path.append('../InternVL/internvl_chat/internvl')

def load_model(model_path):
    """Load InternVL3 model using the official repository."""
    print("Loading InternVL3 model...")
    
    try:
        from internvl.model.internvl_chat import InternVLChatModel
        from internvl.conversation import conv_templates
        from transformers import AutoTokenizer
        
        # Print available keys for debugging
        print("Available conv_templates keys:", list(conv_templates.keys()))
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL3-8B", 
            trust_remote_code=True, 
            use_fast=False
        )
        
        # Initialize the model with minimal dependencies
        model = InternVLChatModel.from_pretrained(
            "OpenGVLab/InternVL3-8B",
            torch_dtype=None,
            device_map="cpu",
            trust_remote_code=True,
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        # Add tokenizer to model for convenience
        model.tokenizer = tokenizer
        
        # Print img_context_token_id for debugging
        print("img_context_token_id:", getattr(model, "img_context_token_id", None))
        print("num_image_token:", getattr(model, "num_image_token", None))
        print("downsample_ratio:", getattr(model, "downsample_ratio", None))
        
        print("InternVL model loaded successfully!")
        return model, conv_templates
        
    except Exception as e:
        print(f"Error loading InternVL model: {e}")
        traceback.print_exc()
        return None, None

def extract_frames_from_directory(video_uid, frame_indices, frames_dir="ego4d_aks_full/frames"):
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

def caption_frame(model, conv_templates, image):
    """Caption a single frame using InternVL."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert image to tensor and add batch dimension
        # Use 448x448 to match the model's expected input size
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Use the chat method instead of generate directly
        response = model.chat(
            tokenizer=model.tokenizer,
            pixel_values=image_tensor,
            question="Describe this image briefly.",
            generation_config={
                'max_new_tokens': 50,
                'do_sample': False,
                'temperature': 0.0
            },
            num_patches_list=[1]  # Single image
        )
        
        # Debug: print the response
        print(f"Generated caption: {response}")
        
        return response
        
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def summarize_captions(model, conv_templates, captions):
    """Create concise summary from frame captions."""
    if not captions:
        return "No captions available."
    
    try:
        # Join captions
        context = " ".join(captions)
        
        # Use the chat method for text-only generation (no image)
        response = model.chat(
            tokenizer=model.tokenizer,
            pixel_values=None,  # No image for summarization
            question=f"Summarize these frame descriptions in one short sentence: {context}",
            generation_config={
                'max_new_tokens': 100,
                'do_sample': False,
                'temperature': 0.0
            },
            num_patches_list=[]  # No images for summarization
        )
        
        return response
        
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
    """Load already processed video_uids from the output JSONL file."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed.add(obj['video_uid'])
                except Exception:
                    continue
    return processed

def process_ego4d_aks_videos(aks_file, model, conv_templates, output_file, max_videos=None):
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
        original_video_path = video_data['video_path']
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
            if existing_result and len(existing_result.get('frame_captions', [])) == len(selected_frames):
                print(f"Video {video_uid} already has complete frame captions, skipping...")
                continue
            
            # Extract frames from directory
            frames = extract_frames_from_directory(video_uid, selected_frames)
            print(f"Extracted {len(frames)} frames")
            
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            
            # Initialize or get existing frame captions
            if existing_result and 'frame_captions' in existing_result:
                frame_captions = existing_result['frame_captions']
                print(f"Resuming from {len(frame_captions)} existing captions")
            else:
                frame_captions = []
            
            # Caption only missing frames
            for i, frame in enumerate(tqdm(frames, desc="Captioning frames")):
                # Skip if caption already exists and is not an error
                if i < len(frame_captions) and not frame_captions[i].startswith("Error:"):
                    print(f"Frame {i}: Skipping (already captioned)")
                    continue
                
                caption = caption_frame(model, conv_templates, frame)
                
                # Ensure frame_captions list is long enough
                while len(frame_captions) <= i:
                    frame_captions.append("")
                frame_captions[i] = caption
                
                print(f"Frame {i}: {caption[:100]}...")  # Debug output
                
            # Create video summary only after all frames are captioned
            if len(frame_captions) == len(selected_frames) and all(cap and not cap.startswith("Error:") for cap in frame_captions):
                video_summary = summarize_captions(model, conv_templates, frame_captions)
                
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
            else:
                print(f"Video {video_uid} has incomplete or error captions, will resume later")
            
        except Exception as e:
            print(f"Error processing video {video_uid}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully processed {processed_count} new videos. Total results: {len(all_results)}")
    print(f"Results saved to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="InternVL3 Ego4D AKS Frame Captioning")
    parser.add_argument('--aks_results', type=str, required=True, help='ego4d_aks_results.json file')
    parser.add_argument('--output', type=str, default='ego4d_aks_captions.json', help='Output JSON file')
    parser.add_argument('--model_path', type=str, default='models/InternVL3-8B', help='Path to InternVL3 model')
    parser.add_argument('--max_videos', type=int, help='Maximum number of videos to process (for testing)')
    # parser.add_argument('--clear_output', action='store_true', help='Clear existing output file before processing')
    
    args = parser.parse_args()
    
    # Load model
    model, conv_templates = load_model(args.model_path)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Clear output file only if requested
    if args.clear_output:
        clear_output_file(args.output)
    
    # Process videos
    process_ego4d_aks_videos(args.aks_results, model, conv_templates, args.output, args.max_videos)

if __name__ == "__main__":
    main() 