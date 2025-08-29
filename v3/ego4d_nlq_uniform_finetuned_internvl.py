#!/usr/bin/env python3
"""
InternVL3 Ego4D NLQ Validation Uniform Frame Captioning and Summarization (Finetuned Model)
Extracts frames directly from video files and samples 32 frames uniformly from each video.
Uses our finetuned InternVL3 model for captioning and summarization.
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
    """Load the finetuned model with HuggingFace."""
    print("Loading finetuned InternVL3 model...")
    model_path = '../v8/work_dirs/internvl3_8b_single_gpu_aggressive'
    
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return None, None, None
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print("Loading base model architecture...")
        # Load the base model to get the correct architecture
        # Disable flash attention to avoid CUDA kernel errors
        model = AutoModelForCausalLM.from_pretrained(
            "OpenGVLab/InternVL3-8B",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"  # Disable flash attention
        )
        
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-8B", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-8B", trust_remote_code=True)
        
        print("✅ Base model architecture loaded successfully!")
        
        # Now load the finetuned weights
        print("Loading finetuned weights...")
        from safetensors import safe_open
        
        finetuned_state_dict = {}
        for i in range(1, 5):
            safetensor_path = os.path.join(model_path, f"model-{i:05d}-of-00004.safetensors")
            if os.path.exists(safetensor_path):
                print(f"Loading weights from {safetensor_path}")
                with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        finetuned_state_dict[key] = f.get_tensor(key)
        
        if not finetuned_state_dict:
            print("No safetensors files found!")
            return None, None, None
        
        # Load the finetuned weights
        model.load_state_dict(finetuned_state_dict, strict=False)
        
        # Disable flash attention in the vision model as well
        if hasattr(model, 'vision_model'):
            for module in model.vision_model.modules():
                if hasattr(module, 'use_flash_attn'):
                    module.use_flash_attn = False
                if hasattr(module, 'attn_implementation'):
                    module.attn_implementation = 'eager'
        
        # Also disable flash attention in the LLM model
        if hasattr(model, 'llm_model'):
            for module in model.llm_model.modules():
                if hasattr(module, 'use_flash_attn'):
                    module.use_flash_attn = False
                if hasattr(module, 'attn_implementation'):
                    module.attn_implementation = 'eager'
        
        # Disable flash attention globally
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        # Additional flash attention disabling
        os.environ["FLASH_ATTENTION_DISABLE"] = "1"
        os.environ["TORCH_USE_CUDA_DSA"] = "0"
        
        # Set environment variables to disable flash attention
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        print("✅ Finetuned weights loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        
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

def caption_frame(model, tokenizer, processor, image):
    """Caption a single frame using our finetuned model."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare the prompt
        prompt = "Describe this image briefly."
        
        # Set img_context_token_id on the model
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            model.img_context_token_id = 151643
            print(f"Set img_context_token_id to: {model.img_context_token_id}")
        
        # Process the image using torchvision transforms
        from torchvision import transforms
        
        # Use the same transforms as the model expects (448x448)
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process the image
        pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Tokenize the prompt
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Move everything to the same device as the model
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                pixel_values=pixel_values,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (remove the input prompt)
        if prompt in response:
            caption = response.split(prompt)[-1].strip()
        else:
            caption = response.strip()
        
        print(f"Generated caption: {caption}")
        return caption
        
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def summarize_captions(model, tokenizer, processor, captions):
    """Create concise summary from frame captions using our finetuned model."""
    if not captions:
        return "No captions available."
    try:
        context = " ".join(captions)
        prompt = f"Summarize the main activity in these frame descriptions in one short sentence: {context}"
        
        # Set img_context_token_id on the model for text-only generation
        if hasattr(model, 'img_context_token_id') and model.img_context_token_id is None:
            model.img_context_token_id = 151643
            print(f"Set img_context_token_id to: {model.img_context_token_id}")
        
        # Process the input (text only for summarization)
        inputs = processor(
            text=prompt,
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (remove the input prompt)
        if prompt in response:
            summary = response.split(prompt)[-1].strip()
        else:
            summary = response.strip()
        
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

def process_nlq_videos_uniform(model, tokenizer, processor, output_file, max_videos=None, num_frames=32):
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
                caption = caption_frame(model, tokenizer, processor, frame)
                frame_captions.append(caption)
                print(f"Frame {j}: {caption[:100]}...")
            
            # Create video summary
            video_summary = summarize_captions(model, tokenizer, processor, frame_captions)
            
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
    print("InternVL3 Ego4D NLQ Validation Uniform Frame Captioning (Finetuned Model)")
    print("=" * 80)
    print(f"Output file: ego4d_nlq_uniform_finetuned_captions.json")
    print(f"Max videos: All")
    print(f"Frames per video: 32")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load model
    model, tokenizer, processor = load_finetuned_model()
    if not model or not tokenizer or not processor:
        print("Failed to load model. Exiting.")
        return
    
    # Process videos
    output_file = "ego4d_nlq_uniform_finetuned_captions.json"
    results = process_nlq_videos_uniform(model, tokenizer, processor, output_file)
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 