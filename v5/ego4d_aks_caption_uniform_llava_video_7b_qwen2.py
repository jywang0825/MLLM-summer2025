#!/usr/bin/env python3
"""
LLaVA-Video-7B-Qwen2 Ego4D Uniform Frame Captioning and Summarization
Samples frames uniformly from each video and generates captions for those frames.
Uses LLaVA-Video-7B-Qwen2 with standard transformers approach.
Based on: https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2#%23use
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
import gc
import copy

# Disable FlashAttention2 to avoid compatibility issues
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTN_DISABLE"] = "1"

warnings.filterwarnings("ignore")

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleaned up")

def check_gpu_processes():
    """Check what processes are using GPU memory."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("GPU processes:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
        else:
            print("No GPU processes found")
    except Exception as e:
        print(f"Could not check GPU processes: {e}")

def find_available_gpu():
    """Force use of GPU 6 regardless of other processes."""
    if not torch.cuda.is_available():
        return None
    
    # When CUDA_VISIBLE_DEVICES=6 is set, we only see one GPU (index 0)
    # So we always use GPU 0 (which is actually GPU 6)
    target_gpu = 0
    
    torch.cuda.set_device(target_gpu)
    torch.cuda.empty_cache()
    free_memory = torch.cuda.get_device_properties(target_gpu).total_memory - torch.cuda.memory_allocated(target_gpu)
    print(f"GPU {target_gpu} (actual GPU 6): {free_memory / 1e9:.1f} GB free")
    
    # Force use GPU 6 even if memory is limited
    print(f"FORCING use of GPU {target_gpu} (actual GPU 6) with {free_memory / 1e9:.1f} GB free memory")
    print(f"Note: Other processes may be using this GPU, but we'll try to allocate memory anyway")
    return target_gpu

def load_model_standard():
    """Load LLaVA-Video-7B-Qwen2 model using LLaVA-NeXT approach."""
    print("Loading LLaVA-Video-7B-Qwen2 model with LLaVA-NeXT...")
    
    # Clean up memory first
    cleanup_gpu_memory()
    
    # Check GPU availability and find best GPU
    device = "cpu"
    
    if torch.cuda.is_available():
        best_gpu = find_available_gpu()
        if best_gpu is not None:
            # Force use GPU 6 regardless of memory constraints
            device = "cuda"
            # Set CUDA device explicitly
            torch.cuda.set_device(best_gpu)
            # Set environment variables to force GPU selection
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"Using GPU: {torch.cuda.get_device_name(best_gpu)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(best_gpu).total_memory / 1e9:.1f} GB")
            print(f"WARNING: Forcing GPU usage even with limited memory. This may cause OOM errors.")
        else:
            print("No suitable GPU found, falling back to CPU")
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
        device_map = "auto" if device == "cuda" else None
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map, attn_implementation="eager"
        )
        model.eval()
        
        print("LLaVA-Video-7B-Qwen2 model loaded successfully!")
        return tokenizer, model, image_processor
        
    except Exception as e:
        print(f"Error loading LLaVA-Video-7B-Qwen2 model: {e}")
        traceback.print_exc()
        return None, None, None

def load_video(video_path, max_frames_num=32, fps=1, force_sample=True):
    """Load video with uniform frame sampling based on Hugging Face documentation."""
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0
    
    try:
        from decord import VideoReader, cpu
        
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        
        return spare_frames, frame_time_str, video_time
        
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        traceback.print_exc()
        return np.zeros((1, 336, 336, 3)), "0.00s", 0.0

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

def caption_video_standard(tokenizer, model, image_processor, video_path, max_frames_num=32):
    """Caption video using LLaVA-Video-7B-Qwen2 with uniform frame sampling."""
    try:
        # Import required modules
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        import copy
        
        # Load video with uniform sampling
        video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        
        # Preprocess video
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        if torch.cuda.is_available():
            video = video.cuda().bfloat16()  # Use bfloat16 to match model dtype
        video = [video]
        
        # Create conversation template
        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
        
        # Use the image token directly from constants
        question = f"<image>\n{time_instruction}\nPlease describe this video in detail."
        conv = copy.deepcopy(conv_templates[conv_template])
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
        
        # Clean up memory
        del input_ids, cont, video
        cleanup_gpu_memory()
        
        return text_outputs, frame_time, video_time
        
    except Exception as e:
        print(f"Error in caption_video_standard: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}", "0.00s", 0.0

def summarize_captions_standard(tokenizer, model, image_processor, captions):
    """Create concise summary from frame captions using LLaVA-Video-7B-Qwen2."""
    if not captions:
        return "No captions available."
    
    try:
        # Import required modules
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        import copy
        
        # Create a simple text-only prompt for summarization
        context = " ".join(captions)
        prompt = f"Summarize the main activity in these frame descriptions in one short sentence: {context}"
        
        # Create conversation template for text-only generation
        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Generate response
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                do_sample=False,
                temperature=0,
                max_new_tokens=64,
            )
        
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        # Clean up memory
        del input_ids, cont
        cleanup_gpu_memory()
        
        return text_outputs
        
    except Exception as e:
        print(f"Error in summarize_captions_standard: {e}")
        traceback.print_exc()
        return f"Summary error: {str(e)}"

def process_nlq_videos_uniform(tokenizer, model, image_processor, output_file, max_videos=None, num_frames=32):
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
            # Generate video caption using uniform frame sampling
            video_caption, frame_time, video_time = caption_video_standard(
                tokenizer, model, image_processor, video_path, num_frames
            )
            
            # Create result entry
            result = {
                'video_uid': video_uid,
                'video_path': video_path,
                'original_summary': original_summary,
                'uniform_sampled_frames': num_frames,
                'frame_time': frame_time,
                'video_time': video_time,
                'generated_caption': video_caption,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Save incrementally
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Completed video {video_uid}, saved to {output_file}")
            print(f"Generated caption: {video_caption[:200]}...")
            
            # Clean up memory after each video
            cleanup_gpu_memory()
            
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
            logging.FileHandler('ego4d_captioning_llava_video.log'),
            logging.StreamHandler()
        ]
    )
    
    output_file = "ego4d_uniform_captions_llava_video_7b_qwen2.json"
    max_videos = None
    num_frames = 32
    
    print("LLaVA-Video-7B-Qwen2 Ego4D Uniform Frame Captioning")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All'}")
    print(f"Frames per video: {num_frames}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check GPU memory before loading model
    if torch.cuda.is_available():
        print("\nGPU Memory Status:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free = props.total_memory - allocated
            print(f"GPU {i} ({props.name}): {allocated/1e9:.1f}GB allocated, {free/1e9:.1f}GB free, {props.total_memory/1e9:.1f}GB total")
        
        # Check what processes are using GPU memory
        check_gpu_processes()
    
    tokenizer, model, image_processor = load_model_standard()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process videos
    results = process_nlq_videos_uniform(tokenizer, model, image_processor, output_file, max_videos, num_frames)
    
    print(f"\nProcessing completed!")
    print(f"Total videos processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main() 