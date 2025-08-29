#!/usr/bin/env python3
"""
Benchmark InternVL3 model on video captioning tasks using Ego4D data.
This script evaluates the model's performance on VideoRecap and NLQ tasks.
"""

import json
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
from PIL import Image
import cv2
from decord import VideoReader, cpu

# Import InternVL3 model
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from internvl.model.internvl3 import InternVL3ForCausalLM
    from internvl.model.internvl3 import InternVL3Config
except ImportError:
    print("Warning: InternVL3 not found. Please install it first.")
    print("pip install git+https://github.com/OpenGVLab/InternVL.git")

def load_internvl3_model(model_path: str, device: str = 'cuda'):
    """Load InternVL3 model for video captioning."""
    try:
        # Load model configuration
        config = InternVL3Config.from_pretrained(model_path)
        
        # Load model
        model = InternVL3ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading InternVL3 model: {e}")
        return None, None

def extract_video_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Extract frames from video for processing."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            frame = vr[idx]
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        return frames
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def generate_caption_internvl3(model, tokenizer, frames: List[Image.Image], 
                              prompt: str = "Please describe this video in detail:") -> str:
    """Generate caption using InternVL3 model."""
    try:
        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Add video frames to inputs
        inputs['images'] = frames
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if prompt in caption:
            caption = caption.replace(prompt, "").strip()
        
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""

def evaluate_videorecap(model, tokenizer, test_data: List[Dict], 
                       video_dir: str, output_file: str) -> Dict:
    """Evaluate model on VideoRecap task."""
    results = []
    
    for sample in tqdm(test_data, desc="Evaluating VideoRecap"):
        video_uid = sample['video_uid']
        ground_truth = sample['recap']
        
        # Find video file
        video_path = os.path.join(video_dir, f"{video_uid}.mp4")
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        # Extract frames
        frames = extract_video_frames(video_path)
        if not frames:
            continue
        
        # Generate caption
        predicted_caption = generate_caption_internvl3(model, tokenizer, frames)
        
        # Store result
        result = {
            'video_uid': video_uid,
            'ground_truth': ground_truth,
            'predicted': predicted_caption,
            'start_time': sample.get('start_time', 0),
            'end_time': sample.get('end_time', 0)
        }
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'total_samples': len(results),
        'output_file': output_file
    }

def evaluate_nlq(model, tokenizer, test_data: List[Dict], 
                video_dir: str, output_file: str) -> Dict:
    """Evaluate model on NLQ task."""
    results = []
    
    for sample in tqdm(test_data, desc="Evaluating NLQ"):
        video_uid = sample['video_uid']
        query = sample['query']
        start_time = sample['start_time']
        end_time = sample['end_time']
        
        # Find video file
        video_path = os.path.join(video_dir, f"{video_uid}.mp4")
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        
        # Extract frames from the specific time range
        frames = extract_video_frames_from_timerange(video_path, start_time, end_time)
        if not frames:
            continue
        
        # Generate answer
        prompt = f"Question: {query}\nAnswer:"
        predicted_answer = generate_caption_internvl3(model, tokenizer, frames, prompt)
        
        # Store result
        result = {
            'video_uid': video_uid,
            'query': query,
            'predicted_answer': predicted_answer,
            'start_time': start_time,
            'end_time': end_time
        }
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'total_samples': len(results),
        'output_file': output_file
    }

def extract_video_frames_from_timerange(video_path: str, start_time: float, 
                                       end_time: float, num_frames: int = 8) -> List[Image.Image]:
    """Extract frames from a specific time range in the video."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        if start_frame >= len(vr) or end_frame > len(vr):
            return []
        
        # Sample frames from the time range
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            if idx < len(vr):
                frame = vr[idx]
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        return frames
    except Exception as e:
        print(f"Error extracting frames from time range: {e}")
        return []

def calculate_metrics(results: List[Dict], task: str) -> Dict:
    """Calculate evaluation metrics."""
    # This is a placeholder for actual metric calculation
    # You would typically use metrics like BLEU, METEOR, ROUGE, etc.
    
    metrics = {
        'task': task,
        'total_samples': len(results),
        'avg_prediction_length': np.mean([len(r.get('predicted', '')) for r in results]),
        'avg_ground_truth_length': np.mean([len(r.get('ground_truth', '')) for r in results])
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Benchmark InternVL3 on Ego4D tasks')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to InternVL3 model')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSON file')
    parser.add_argument('--task', type=str, choices=['videorecap', 'nlq'], required=True,
                       help='Task to evaluate')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading InternVL3 model from {args.model_path}")
    model, tokenizer = load_internvl3_model(args.model_path, args.device)
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Evaluate based on task
    if args.task == 'videorecap':
        output_file = os.path.join(args.output_dir, 'videorecap_results.json')
        results = evaluate_videorecap(model, tokenizer, test_data, args.video_dir, output_file)
    elif args.task == 'nlq':
        output_file = os.path.join(args.output_dir, 'nlq_results.json')
        results = evaluate_nlq(model, tokenizer, test_data, args.video_dir, output_file)
    
    # Calculate metrics
    with open(output_file, 'r') as f:
        evaluation_results = json.load(f)
    
    metrics = calculate_metrics(evaluation_results, args.task)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, f'{args.task}_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation completed!")
    print(f"Results saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Total samples evaluated: {metrics['total_samples']}")

if __name__ == "__main__":
    main()