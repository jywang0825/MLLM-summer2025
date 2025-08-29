#!/usr/bin/env python3
"""
Intern2.5-8B Ego4D AKS Frame Captioning and Summarization (GPU Optimized)
Extracts frames at indices specified in selected_frames from each video and generates captions for those frames.
Uses Intern2.5-8B for proper GPU inference.
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
sys.path.append("../models/InternVL2_5-8B")
# from conversation import get_conv_template  # Not needed for this implementation

def load_model():
    """Load Intern2.5-8B model for GPU inference."""
    print("Loading Intern2.5-8B model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        
        # Load model from local path
        model_path = "../models/InternVL2_5-8B"
        print(f"Loading model from: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load the model using AutoModelForCausalLM to ensure generation capabilities
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Disable FlashAttention in all ViT layers
        for module in model.modules():
            if hasattr(module, "use_flash_attn"):
                module.use_flash_attn = False

        print("Intern2.5-8B model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def find_video_path_from_aks(video_path):
    """Map the AKS absolute video path to the local path."""
    if video_path.startswith("/shared/ssd_14T/home/wangj/remote_ego4d/"):
        relative_path = video_path.replace("/shared/ssd_14T/home/wangj/remote_ego4d/", "")
        local_path = f"../remote_ego4d/{relative_path}"
        return local_path
    else:
        return video_path

def extract_selected_frames_from_video(video_path, frame_indices):
    """Extract specific frames from a video file using frame indices."""
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
        print(f"Requested frame indices: {frame_indices[:5]}... (total: {len(frame_indices)})")
        frames = []
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= total_frames:
                print(f"Frame index {frame_idx} out of bounds (>= {total_frames}), skipping...")
                continue
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

def caption_frame(model, tokenizer, image):
    """Caption a single frame using Intern2.5-8B model."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        prompt = "Describe this image briefly."
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        device = next(model.parameters()).device
        pixel_values = transform(image).unsqueeze(0).to(device)
        pixel_values = pixel_values.half()
        generation_config = {
            'max_new_tokens': 100,
            'do_sample': True,
            'temperature': 0.7
        }
        try:
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config,
                verbose=True
            )
        except ValueError as ve:
            # If the error is about unpacking, use batch_chat instead
            if 'not enough values to unpack' in str(ve):
                responses = model.batch_chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    questions=[prompt],
                    generation_config=generation_config,
                    verbose=True
                )
                response = responses[0]  # Get the first (and only) response
            else:
                raise
        return response.strip()
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return "Error generating caption"

def summarize_captions(model, tokenizer, captions):
    """Summarize the captions using Intern2.5-8B model."""
    try:
        summary_prompt = f"Summarize these image captions in one sentence: {' '.join(captions)}"
        device = next(model.parameters()).device
        dummy_image = torch.zeros(1, 3, 448, 448, dtype=torch.float16, device=device)
        generation_config = {
            'max_new_tokens': 100,
            'do_sample': True,
            'temperature': 0.7
        }
        try:
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=dummy_image,
                question=summary_prompt,
                generation_config=generation_config,
                verbose=True
            )
        except ValueError as ve:
            if 'not enough values to unpack' in str(ve):
                responses = model.batch_chat(
                    tokenizer=tokenizer,
                    pixel_values=dummy_image,
                    questions=[summary_prompt],
                    generation_config=generation_config,
                    verbose=True
                )
                response = responses[0]  # Get the first (and only) response
            else:
                raise
        return response.strip()
    except Exception as e:
        print(f"Error in summarize_captions: {e}")
        traceback.print_exc()
        return "Error generating summary"

def process_aks_videos(model, tokenizer, aks_file, output_file, max_videos=None):
    """Process AKS videos using selected_frames from AKS JSON."""
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
    existing_uids = {result['video_uid'] for result in existing_results}
    all_results = existing_results.copy()
    for i, video_info in enumerate(tqdm(aks_results, desc="Processing videos")):
        video_uid = video_info['video_uid']
        original_video_path = video_info['video_path']
        original_summary = video_info.get('summary', '')
        selected_frames = video_info.get('selected_frames', [])
        if video_uid in existing_uids:
            print(f"Video {video_uid} already processed, skipping...")
            continue
        print(f"\nProcessing video {i+1}/{len(aks_results)}: {video_uid}")
        print(f"Original path: {original_video_path}")
        print(f"Selected frames: {len(selected_frames)}")
        print(f"Sample frames: {selected_frames[:5]}")
        if not selected_frames:
            print(f"No selected frames for video: {video_uid}")
            continue
        video_path = find_video_path_from_aks(original_video_path)
        print(f"Mapped path: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        try:
            frames = extract_selected_frames_from_video(video_path, selected_frames)
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            frame_captions = []
            for j, frame in enumerate(tqdm(frames, desc="Captioning frames")):
                caption = caption_frame(model, tokenizer, frame)
                frame_captions.append(caption)
                print(f"Frame {j}: {caption[:100]}...")
            video_summary = summarize_captions(model, tokenizer, frame_captions)
            result = {
                'video_uid': video_uid,
                'video_path': video_path,
                'original_summary': original_summary,
                'selected_frames': selected_frames,
                'frame_captions': frame_captions,
                'generated_summary': video_summary,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)
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
            logging.FileHandler('ego4d_aks_captioning.log'),
            logging.StreamHandler()
        ]
    )
    aks_file = "../v1/ego4d_aks_full/ego4d_aks_results.json"
    output_file = "ego4d_aks_captions_intern2.5_8b.json"
    max_videos = 2  # Test with two videos to get a fresh one
    print("Intern2.5-8B Ego4D AKS Frame Captioning (selected frames)")
    print("=" * 60)
    print(f"AKS results file: {aks_file}")
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    model_components = load_model()
    if not model_components:
        print("Failed to load model. Exiting.")
        return
    model, tokenizer = model_components
    results = process_aks_videos(model, tokenizer, aks_file, output_file, max_videos)
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 