#!/usr/bin/env python3
"""
InternVL3 Ego4D NLQ Validation Uniform Frame Captioning (Finetuned Model - Simple Approach)
Builds on our working finetuned model that successfully generates text.
"""
# Disable flash attention before any imports
import os
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

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

# Add the custom model code to the path
sys.path.append('/shared/ssd_14T/home/wangj/your-repo/InternVL/internvl_chat/internvl/model/internvl_chat')
# Add the main InternVL path for imports
sys.path.append('/shared/ssd_14T/home/wangj/your-repo/InternVL')

def load_finetuned_model():
    """Load the finetuned model with HuggingFace - using our working approach."""
    print("Loading finetuned InternVL3 model...")
    model_path = '../v8/work_dirs/internvl3_8b_single_gpu_aggressive/checkpoint-11000'
    
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return None, None, None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print("Loading finetuned model directly...")
        # Load the finetuned model directly from the checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Explicitly set to bfloat16
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Disable flash attention
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print("✅ Finetuned model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None, None, None

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
        
        cap.release()
        print(f"Extracted {len(frames)} frames")
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def test_text_generation(model, tokenizer):
    """Test simple text generation with the finetuned model - our working approach."""
    try:
        print("Testing text generation...")
        
        # Set the image context token ID BEFORE any generation
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            model.img_context_token_id = 151643
            print(f"Set img_context_token_id to: {model.img_context_token_id}")
        
        # Simple text generation test
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device and dtype as the model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # input_ids should remain Long, only attention_mask gets converted to model dtype
        inputs = {
            'input_ids': inputs['input_ids'].to(device=device, dtype=torch.long),
            'attention_mask': inputs['attention_mask'].to(device=device, dtype=dtype)
        }
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Text generation test passed!")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
        return True
        
    except Exception as e:
        print(f"Error in text generation: {e}")
        traceback.print_exc()
        return False

def caption_frame_simple(model, tokenizer, image):
    """Caption a single frame using our finetuned model with proper image processing."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Set the image context token ID BEFORE any generation
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            model.img_context_token_id = 151643
            print(f"Set img_context_token_id to: {model.img_context_token_id}")
        
        # Process the image using torchvision transforms (448x448 as per model config)
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process the image and convert to model dtype
        pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Convert to bfloat16 immediately after processing
        pixel_values = pixel_values.to(torch.bfloat16)
        
        # Tokenize the prompt
        prompt = "Describe what you see in this image."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move everything to the same device and dtype as the model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Convert pixel_values to model dtype and move to device
        # Force bfloat16 to match model expectations
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        
        # input_ids should remain Long, only attention_mask gets converted to model dtype
        inputs = {
            'input_ids': inputs['input_ids'].to(device=device, dtype=torch.long),
            'attention_mask': inputs['attention_mask'].to(device=device, dtype=dtype)
        }
        
        # Use the model's chat method which handles data types better
        with torch.no_grad():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question="Describe what you see in this image concisely in one sentence.",
                generation_config={
                    'max_new_tokens': 100,  # Increased from 50 to allow for more detailed descriptions
                    'do_sample': True,
                    'temperature': 0.7,
                    'pad_token_id': tokenizer.eos_token_id
                },
                num_patches_list=[1]  # Single image
            )
        
        print(f"Generated caption: {response}")
        return response
        
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def summarize_captions_simple(model, tokenizer, captions):
    """Create concise summary from frame captions using our working approach."""
    if not captions:
        return "No captions available."
    try:
        context = " ".join(captions)
        prompt = f"Write exactly one sentence describing the activity in the video. Keep it short and concise: {context}"
        
        # Set the image context token ID BEFORE any generation (exact same as working code)
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            model.img_context_token_id = 151643
            print(f"Set img_context_token_id to: {model.img_context_token_id}")
        
        # Simple text generation test (exact same as working code)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device and dtype as the model (exact same as working code)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # input_ids should remain Long, only attention_mask gets converted to model dtype
        inputs = {
            'input_ids': inputs['input_ids'].to(device=device, dtype=torch.long),
            'attention_mask': inputs['attention_mask'].to(device=device, dtype=dtype)
        }
        
        # Generate summary (exact same as working code)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # Set a reasonable limit to avoid the error
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response (exact same as working code)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (remove the input prompt)
        if prompt in generated_text:
            summary = generated_text.split(prompt)[-1].strip()
        else:
            summary = generated_text.strip()
        
        # Post-process to ensure only one sentence
        sentences = summary.split('.')
        if len(sentences) > 1:
            # Take only the first complete sentence
            summary = sentences[0].strip() + '.'
        
        print(f"Generated summary: {summary}")
        return summary
        
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

def process_nlq_videos_uniform_simple(model, tokenizer, output_file, max_videos=None, num_frames=32):
    """Process NLQ validation videos with uniform frame sampling - using our working approach."""
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
            
            # Generate captions for each frame using our working approach
            frame_captions = []
            for j, frame in enumerate(tqdm(frames, desc="Captioning frames")):
                caption = caption_frame_simple(model, tokenizer, frame)
                frame_captions.append(caption)
                print(f"Frame {j}: {caption[:100]}...")
            
            # Create video summary
            video_summary = summarize_captions_simple(model, tokenizer, frame_captions)
            
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
    print("InternVL3 Ego4D NLQ Validation Uniform Frame Captioning (Checkpoint Model - Simple)")
    print("=" * 80)
    print(f"Output file: checkpoint_model_output.json")
    print(f"Max videos: All")
    print(f"Frames per video: 32")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load model
    model, tokenizer, processor = load_finetuned_model()
    if not model or not tokenizer or not processor:
        print("Failed to load model. Exiting.")
        return
    
    # Test the model first
    print("\nTesting model functionality...")
    success = test_text_generation(model, tokenizer)
    if not success:
        print("Model test failed. Exiting.")
        return
    
    print("✅ Model test passed! Proceeding with video processing...")
    
    # Process videos
    output_file = "checkpoint_model_output.json"
    results = process_nlq_videos_uniform_simple(model, tokenizer, output_file)
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 