#!/usr/bin/env python3
"""
LLaVA-Video-7B-Qwen2 Ego4D AKS Frame Captioning and Summarization (GPU Optimized)
Processes ego4d_aks_results.json format and generates captions for AKS-selected frames.
Uses LLaVA-Video-7B-Qwen2 for proper GPU inference.
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
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")

def load_model():
    """Load LLaVA-Video-7B-Qwen2 model for GPU inference."""
    print("Loading LLaVA-Video-7B-Qwen2 model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        # Install required dependencies if not already installed
        # pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
        
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
        from llava.conversation import conv_templates, SeparatorStyle
        
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        model_name = "llava_qwen"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = "auto"
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
        )
        model.eval()
        
        print("LLaVA-Video-7B-Qwen2 model loaded successfully!")
        return tokenizer, model, image_processor
        
    except Exception as e:
        print(f"Error loading LLaVA-Video-7B-Qwen2 model: {e}")
        traceback.print_exc()
        return None, None, None

def extract_frames_from_directory(video_uid, frame_indices, frames_dir="../v1/ego4d_aks_full/frames"):
    """Extract frames from the frames directory using frame indices."""
    frames = []
    video_frames_dir = os.path.join(frames_dir, video_uid)
    
    if not os.path.exists(video_frames_dir):
        print(f"Frames directory not found: {video_frames_dir}")
        return frames
    
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        print(f"No frame files found in: {video_frames_dir}")
        return frames
    
    print(f"Found {len(frame_files)} frame files")
    print(f"Frame indices to extract: {frame_indices[:5]}...")  # Show first 5 indices
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx < len(frame_files):
            frame_path = os.path.join(video_frames_dir, frame_files[frame_idx])
            try:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    print(f"Loaded frame {i}: {frame_files[frame_idx]}")
                else:
                    print(f"Failed to load frame {i}: {frame_path}")
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
    
    return frames

def caption_frames_llava(tokenizer, model, image_processor, frames, video_time=0.0):
    """Caption frames using LLaVA-Video-7B-Qwen2."""
    try:
        if not frames:
            return ["No frames available"]
        
        # Convert frames to video format expected by LLaVA-Video
        video = np.array(frames)
        
        # Preprocess video
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        if torch.cuda.is_available():
            video = video.cuda().half()
        video = [video]
        
        # Create conversation template
        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are selected from it. Please describe this video in detail."
        
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruction}"
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Generate response
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
            )
        
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        # Split the response into individual frame descriptions if needed
        # For now, return the full description
        return [text_outputs]
        
    except Exception as e:
        print(f"Error in caption_frames_llava: {e}")
        traceback.print_exc()
        return [f"Error: {str(e)}"]

def summarize_captions_llava(tokenizer, model, image_processor, captions):
    """Create concise summary from frame captions using LLaVA-Video-7B-Qwen2."""
    if not captions:
        return "No captions available."
    
    try:
        context = " ".join(captions)
        prompt = f"Summarize the main activity in these frame descriptions in one short sentence: {context}"
        
        # Create conversation for text-only summarization
        conv_template = "qwen_1_5"
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer(prompt_question, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        
        with torch.no_grad():
            cont = model.generate(
                input_ids.input_ids,
                do_sample=False,
                temperature=0,
                max_new_tokens=128,
            )
        
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
        
    except Exception as e:
        print(f"Error in summarize_captions_llava: {e}")
        traceback.print_exc()
        return f"Summary error: {str(e)}"

def process_ego4d_aks_videos(aks_file, tokenizer, model, image_processor, output_file, max_videos=None):
    with open(aks_file, 'r') as f:
        aks_results = json.load(f)
    print(f"Loaded {len(aks_results)} videos from AKS results")
    
    if max_videos:
        aks_results = aks_results[:max_videos]
        print(f"Processing first {len(aks_results)} videos")
    
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            print(f"Loaded {len(existing_results)} existing results")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = []
    
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
            existing_result = existing_results_map.get(video_uid)
            if existing_result and existing_result.get('generated_summary'):
                print(f"Video {video_uid} already has complete results, skipping...")
                continue
            
            # Extract AKS-selected frames
            frames = extract_frames_from_directory(video_uid, selected_frames)
            print(f"Extracted {len(frames)} frames")
            
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            
            # Caption frames using LLaVA-Video
            frame_captions = caption_frames_llava(tokenizer, model, image_processor, frames)
            print(f"Generated {len(frame_captions)} captions")
            
            # Generate summary
            video_summary = summarize_captions_llava(tokenizer, model, image_processor, frame_captions)
            
            result = {
                'video_uid': video_uid,
                'original_video_path': original_video_path,
                'original_summary': original_summary,
                'selected_frames': selected_frames,
                'sampling_method': 'aks',
                'num_frames': len(frames),
                'frame_captions': frame_captions,
                'generated_summary': video_summary
            }
            
            if video_uid in existing_results_map:
                for j, existing_result in enumerate(all_results):
                    if existing_result['video_uid'] == video_uid:
                        all_results[j] = result
                        break
            else:
                all_results.append(result)
            
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ego4d_captioning_llava_video_aks.log'),
            logging.StreamHandler()
        ]
    )
    
    aks_file = "../v1/ego4d_aks_full/ego4d_aks_results.json"
    output_file = "ego4d_aks_captions_llava_video_7b_qwen2_aks.json"
    max_videos = None
    
    print("LLaVA-Video-7B-Qwen2 Ego4D AKS Frame Captioning")
    print("=" * 60)
    print(f"AKS file: {aks_file}")
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not os.path.exists(aks_file):
        print(f"Error: AKS file not found: {aks_file}")
        return
    
    tokenizer, model, image_processor = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    process_ego4d_aks_videos(aks_file, tokenizer, model, image_processor, output_file, max_videos)
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main() 