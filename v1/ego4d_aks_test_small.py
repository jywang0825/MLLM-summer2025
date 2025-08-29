#!/usr/bin/env python3
"""
Small test version of AKS for Ego4D NLQ Dataset - processes only first 5 videos
"""

import torch
import json
import numpy as np
import os
import argparse
import heapq
from PIL import Image
from decord import VideoReader, cpu
try:
    from lavis.models import load_model_and_preprocess
except ImportError:
    print("Warning: LAVIS not found. Please install it for BLIP support.")
    load_model_and_preprocess = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    from transformers.models.clip.processing_clip import CLIPProcessor
    from transformers.models.clip.modeling_clip import CLIPModel
from tqdm import tqdm

def load_manifest(manifest_path):
    """Load clips manifest to map parent_video_uid to exported_clip_uid and filename."""
    import csv
    parent_to_clips = {}
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent_video_uid = row['parent_video_uid']
            exported_clip_uid = row['exported_clip_uid']
            s3_path = row['s3_path']
            # Extract filename from s3 path
            filename = s3_path.split('/')[-1]
            
            if parent_video_uid not in parent_to_clips:
                parent_to_clips[parent_video_uid] = []
            parent_to_clips[parent_video_uid].append({
                'exported_clip_uid': exported_clip_uid,
                'filename': filename
            })
    
    return parent_to_clips

def load_model(model_name, device):
    """Load the specified model for feature extraction."""
    if model_name == 'blip':
        if load_model_and_preprocess is None:
            raise ValueError("LAVIS not installed. Please install it for BLIP support.")
        model, vis_processors, text_processors = load_model_and_preprocess(
            "blip_image_text_matching", "large", device=device, is_eval=True
        )
        return model, vis_processors, text_processors
    elif model_name == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor, None
    else:
        raise ValueError(f"Model {model_name} not supported")

def extract_features_for_video(video_path, summary, model, vis_processors, text_processors, 
                              model_name, device):
    """Extract features for a single video using the summary as query."""
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return [], []
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # Sample frames at 1 FPS for efficiency
        frame_interval = int(fps)
        frame_nums = total_frames // frame_interval
        
        scores = []
        frame_indices = []
        
        # Use summary as the query text
        query_text = summary
        
        # Limit to first 50 frames for testing
        for j in range(min(frame_nums, 50)):
            frame_idx = j * frame_interval
            if frame_idx >= total_frames:
                break
                
            try:
                raw_image = np.array(vr[frame_idx])
                raw_image = Image.fromarray(raw_image)
                
                if model_name == 'blip':
                    txt = text_processors["eval"](query_text)
                    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        blip_output = model({"image": img, "text_input": txt}, match_head="itm")
                    blip_scores = torch.softmax(blip_output, dim=1)
                    score = blip_scores[:, 1].item()
                    
                elif model_name == 'clip':
                    inputs_text = vis_processors(text=query_text, return_tensors="pt", 
                                               padding=True, truncation=True).to(device)
                    text_features = model.get_text_features(**inputs_text)
                    
                    inputs_image = vis_processors(images=raw_image, return_tensors="pt", 
                                                padding=True).to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**inputs_image)
                    score = torch.nn.CosineSimilarity(dim=-1)(text_features, image_features).item()
                
                scores.append(score)
                frame_indices.append(frame_idx)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
        
        return scores, frame_indices
        
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return [], []

def select_frames_aks(scores, frame_indices, max_num_frames):
    """Apply AKS algorithm to select frames."""
    if len(scores) <= max_num_frames:
        return frame_indices
    
    # Simple top-k selection for testing
    scores = np.array(scores)
    top_indices = np.argsort(scores)[-max_num_frames:]
    selected_frames = [frame_indices[i] for i in top_indices]
    selected_frames.sort()
    return selected_frames

def main():
    # Fixed parameters for testing
    nlq_val_path = './nlq_val_summaries.json'
    video_path = '~/remote_ego4d/v2/clips'
    manifest_path = '~/remote_ego4d/v2/clips/manifest.csv'
    extract_feature_model = 'clip'
    output_dir = './ego4d_aks_test_small'
    max_num_frames = 8
    device = 'cpu'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NLQ validation data (only first 5)
    with open(nlq_val_path, 'r') as f:
        nlq_data = json.load(f)[:5]  # Only first 5 videos
    
    print(f"Testing with {len(nlq_data)} videos")
    
    # Load manifest for video path mapping
    manifest_path = os.path.expanduser(manifest_path)
    parent_to_clips = load_manifest(manifest_path)
    
    # Load model
    model, vis_processors, text_processors = load_model(extract_feature_model, device)
    
    # Process each video
    results = []
    
    for i, item in enumerate(nlq_data):
        video_uid = item['video_uid']
        summary = item['summary']
        
        print(f"Processing video {i+1}/{len(nlq_data)}: {video_uid}")
        
        # Find corresponding video file
        if video_uid in parent_to_clips:
            # Take the first available clip for this parent video
            clip_info = parent_to_clips[video_uid][0]
            video_filename = clip_info['filename']
        else:
            print(f"Could not find video for {video_uid}")
            continue
        
        video_path_full = os.path.join(os.path.expanduser(video_path), video_filename)
        print(f"Video path: {video_path_full}")
        
        # Extract features
        scores, frame_indices = extract_features_for_video(
            video_path_full, summary, model, vis_processors, text_processors,
            extract_feature_model, device
        )
        
        if len(scores) == 0:
            print(f"No features extracted for {video_uid}")
            continue
        
        print(f"Extracted {len(scores)} frame scores")
        
        # Apply simple frame selection
        selected_frames = select_frames_aks(scores, frame_indices, max_num_frames)
        
        # Store results
        result = {
            'video_uid': video_uid,
            'summary': summary,
            'video_path': video_path_full,
            'num_frames_processed': len(scores),
            'selected_frames': selected_frames,
            'num_selected': len(selected_frames)
        }
        results.append(result)
        
        print(f"Selected {len(selected_frames)} frames")
    
    # Save results
    output_file = os.path.join(output_dir, 'ego4d_aks_test_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test completed!")
    print(f"Processed {len(results)} videos")
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main() 