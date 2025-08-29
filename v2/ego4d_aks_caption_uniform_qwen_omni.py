#!/usr/bin/env python3
"""
Qwen Omni Ego4D Uniform Frame Captioning and Summarization (Qwen Omni GPU Optimized)
Samples 32 frames uniformly from each video and generates captions for those frames.
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
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_model():
    """Load Qwen2.5-Omni-7B model for GPU inference."""
    print("Loading Qwen2.5-Omni-7B model...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU available, falling back to CPU")
    
    try:
        # Load model and processor from local directory
        model_path = "./qwen2.5-omni-7b"
        print(f"Loading model from: {model_path}")
        
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": 7},  # Force to GPU 7 which has 42.8GB free
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"  # Use more memory-efficient attention
        )
        
        print("Qwen2.5-Omni-7B model loaded successfully!")
        
        # Enable memory optimizations
        model.gradient_checkpointing_enable()
        model.eval()  # Set to evaluation mode
        
        return model, processor
        
    except Exception as e:
        print(f"Error loading Qwen2.5-Omni-7B model: {e}")
        traceback.print_exc()
        return None, None

def sample_uniform_frames(video_frames_dir, num_frames=32):
    """Sample num_frames uniformly from the available frames in the directory."""
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith('.jpg')])
    total_frames = len(frame_files)
    if total_frames == 0:
        print(f"No frame files found in: {video_frames_dir}")
        return []
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    sampled_files = [frame_files[i] for i in indices]
    return sampled_files

def find_video_path(clip_uid, clips_dir="../remote_ego4d/v2/clips"):
    """Find the video file path for a given clip UID."""
    video_path = os.path.join(clips_dir, f"{clip_uid}.mp4")
    if os.path.exists(video_path):
        return video_path
    else:
        print(f"Video file not found: {video_path}")
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

def caption_frame(model, processor, image):
    """Caption a single frame using Qwen2.5-Omni-7B model."""
    try:
        print("▶️  caption_frame start", flush=True)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image to reduce memory usage
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Create conversation with image
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Provide a brief, factual description of what you see in this image."}
                ]
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        
        print(f"   → inputs ready, calling generate()", flush=True)
        # Generate response with length constraints
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=False, 
            return_audio=False,
            max_new_tokens=64,       # Increased for better responses
            do_sample=True,          # Enable sampling for more natural text
            temperature=0.7,         # Add some randomness
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        print("   → generate() returned", flush=True)
        
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        full_response = response[0]
        print(f"   → DEBUG: Full response: {full_response}", flush=True)
        
        # Extract caption using the same logic as batch processing
        if "assistant" in full_response.lower():
            # Find the assistant response
            assistant_start = full_response.lower().find("assistant")
            if assistant_start != -1:
                # Get everything after "assistant"
                assistant_part = full_response[assistant_start:]
                # Remove "assistant" prefix and clean up
                caption = assistant_part[len("assistant"):].strip()
                
                # Clean up any remaining artifacts
                for prefix in ["system", "user"]:
                    if caption.lower().startswith(prefix):
                        caption = caption[len(prefix):].strip()
                
                if not caption or len(caption) < 5:
                    caption = "No caption generated"
            else:
                caption = "No caption generated"
        else:
            caption = "No caption generated"
        
        print(f"Generated caption: {caption}")
        
        # Clean up memory
        del text_ids
        del inputs
        torch.cuda.empty_cache()
        
        return caption
        
    except Exception as e:
        print(f"Error in caption_frame: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}"

def summarize_captions(model, processor, captions):
    """Create concise summary from frame captions using Qwen2.5-Omni-7B."""
    if not captions:
        return "No captions available."
    
    # Filter out "No caption generated" entries
    valid_captions = [cap for cap in captions if cap != "No caption generated" and cap.strip()]
    if not valid_captions:
        return "No valid captions available for summarization."
    
    print(f"   → Summarizing {len(valid_captions)} valid captions out of {len(captions)} total")
    print(f"   → Sample captions: {valid_captions[:3]}")
    
    try:
        context = " ".join(valid_captions)
        
        # Create conversation for summarization
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": f"Summarize the main activity in one factual sentence based on these frame descriptions: {context}"
            }
        ]
        
        # Process inputs
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        
        print(f"   → Generating summary...", flush=True)
        # Generate response
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=False, 
            return_audio=False,
            max_new_tokens=64,       # Limit the length of summary
            early_stopping=True,     # Stop when EOS token is generated
            do_sample=True,          # Enable sampling for more natural text
            temperature=0.7,         # Add some randomness
            pad_token_id=processor.tokenizer.eos_token_id
        )
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        print(f"   → DEBUG: Raw summary response: {response[0][:200]}...", flush=True)
        
        # Extract the assistant response properly for Qwen2.5-Omni
        full_response = response[0]
        summary = ""
        
        # Method 1: Look for assistant response in the full response
        if "assistant" in full_response.lower():
            # Find the assistant response
            assistant_start = full_response.lower().find("assistant")
            if assistant_start != -1:
                # Get everything after "assistant"
                assistant_part = full_response[assistant_start:]
                # Remove "assistant" prefix and clean up
                summary = assistant_part[len("assistant"):].strip()
                
                # Clean up any remaining artifacts
                for prefix in ["system", "user"]:
                    if summary.lower().startswith(prefix):
                        summary = summary[len(prefix):].strip()
                
                # Remove any remaining conversation artifacts
                if "<|im_end|>" in summary:
                    summary = summary.split("<|im_end|>")[0].strip()
                
                # Remove any remaining prompt text
                if "summarize the main activity" in summary.lower():
                    # Find where the actual summary starts
                    prompt_end = summary.lower().find("summarize the main activity")
                    if prompt_end != -1:
                        # Look for the actual response after the prompt
                        remaining = summary[prompt_end:]
                        # Find the first sentence that doesn't contain the prompt
                        sentences = remaining.split('.')
                        for sentence in sentences:
                            if "summarize the main" not in sentence.lower() and "provide a one-sentence" not in sentence.lower():
                                summary = sentence.strip()
                                break
        
        # Method 2: If assistant method failed, try to extract after the last <|im_end|>
        if not summary or len(summary) < 10:
            parts = full_response.split('<|im_end|>')
            if len(parts) > 1:
                summary = parts[-1].strip()
            else:
                summary = full_response.strip()
        
        # Method 3: If still no good summary, try to find the last meaningful sentence
        if not summary or len(summary) < 10 or "user" in summary.lower():
            # Split by sentences and find the last meaningful one
            sentences = full_response.split('.')
            for sentence in reversed(sentences):
                sentence = sentence.strip()
                if (len(sentence) > 10 and 
                    "user" not in sentence.lower() and 
                    "system" not in sentence.lower() and
                    "assistant" not in sentence.lower() and
                    "summarize the main" not in sentence.lower()):
                    summary = sentence
                    break
        
        # Final cleanup and validation
        if not summary or len(summary) < 10:
            summary = "Summary generation failed"
        
        # Additional safety check: if the summary is too long or contains conversation elements, extract just the first sentence
        if len(summary) > 200 or "user" in summary.lower() or "system" in summary.lower():
            # Try to extract just the first meaningful sentence
            sentences = summary.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 10 and 
                    "user" not in sentence.lower() and 
                    "system" not in sentence.lower() and
                    "assistant" not in sentence.lower() and
                    "summarize the main" not in sentence.lower()):
                    summary = sentence
                    break
            else:
                summary = "Summary extraction failed"
        
        # Clean the summary of conversational elements
        cleaned_summary = clean_conversational_elements(summary)
        
        print(f"   → Final summary: {cleaned_summary}", flush=True)
        
        # Clean up memory
        del text_ids
        del inputs
        torch.cuda.empty_cache()
        
        return cleaned_summary
        
    except Exception as e:
        print(f"Error in summarize_captions: {e}")
        traceback.print_exc()
        return f"Summary error: {str(e)}"

def caption_frames_batch(model, processor, frames):
    """Caption multiple frames in a single batch for speed."""
    try:
        print("▶️  caption_frames_batch start", flush=True)
        if not frames:
            return []
        
        # Convert frames to PIL Images
        images = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(frame)
            else:
                pil_image = frame
            
            # Resize image to reduce memory usage
            pil_image = pil_image.resize((224, 224), Image.LANCZOS)
            images.append(pil_image)
        
        # Create batch conversation
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            }
        ]
        
        # Add each image with a prompt
        for i, image in enumerate(images):
            conversation.append(                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image factually in one sentence. Do not ask questions or use conversational language."}
                    ]
                })
        
        # Process inputs
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
        inputs = inputs.to(model.device).to(model.dtype)
        
        print(f"   → inputs ready, calling generate() for {len(frames)} frames", flush=True)
        # Generate response with length constraints
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=False, 
            return_audio=False,
            max_new_tokens=64,       # Increased for better responses
            do_sample=True,          # Enable sampling for more natural text
            temperature=0.7,         # Add some randomness
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
        print("   → generate() returned", flush=True)
        
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Clean up memory
        del text_ids
        del inputs
        torch.cuda.empty_cache()
        
        # Extract individual captions
        full_response = response[0]
        print(f"   → DEBUG: Full response: {full_response[:200]}...", flush=True)
        captions = []
        
        # For single frame, extract the response after the last <|im_end|>
        if len(frames) == 1:
            print(f"   → DEBUG: Full response: {full_response}", flush=True)
            
            # According to Qwen2.5-Omni docs, extract the assistant response
            # The response should contain the assistant's reply after the user prompt
            if "assistant" in full_response.lower():
                # Find the assistant response
                assistant_start = full_response.lower().find("assistant")
                if assistant_start != -1:
                    # Get everything after "assistant"
                    assistant_part = full_response[assistant_start:]
                    # Remove "assistant" prefix and clean up
                    response_text = assistant_part[len("assistant"):].strip()
                    
                    # Clean up any remaining artifacts
                    for prefix in ["system", "user"]:
                        if response_text.lower().startswith(prefix):
                            response_text = response_text[len(prefix):].strip()
                    
                    if response_text and len(response_text) > 5:
                        cleaned_caption = clean_conversational_elements(response_text)
                        captions.append(cleaned_caption)
                        print(f"   → DEBUG: Extracted caption: {cleaned_caption[:50]}...", flush=True)
                    else:
                        captions.append("No caption generated")
                        print(f"   → DEBUG: No substantial response found", flush=True)
                else:
                    captions.append("No caption generated")
                    print(f"   → DEBUG: Assistant response not found", flush=True)
            else:
                # Try alternative extraction - look for response after the prompt
                user_prompt = "Describe this image briefly in one sentence."
                if user_prompt in full_response:
                    response_part = full_response.split(user_prompt)[-1].strip()
                    print(f"   → DEBUG: Response after prompt: {response_part[:100]}...", flush=True)
                    
                    # Clean up any remaining prompt artifacts
                    cleaned = response_part
                    for prefix in ["assistant", "system", "user"]:
                        if cleaned.lower().startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                    
                    if cleaned and len(cleaned) > 10:  # Must be substantial
                        cleaned_caption = clean_conversational_elements(cleaned)
                        captions.append(cleaned_caption)
                        print(f"   → DEBUG: Extracted caption: {cleaned_caption[:50]}...", flush=True)
                    else:
                        captions.append("No caption generated")
                        print(f"   → DEBUG: No substantial response found", flush=True)
                else:
                    captions.append("No caption generated")
                    print(f"   → DEBUG: User prompt not found in response", flush=True)
        else:
            # For multiple frames, split by <|im_end|> and extract each response
            parts = full_response.split('<|im_end|>')
            print(f"   → DEBUG: Found {len(parts)} parts after splitting", flush=True)
            
            for i, part in enumerate(parts[1:]):  # Skip the first part (before first image)
                print(f"   → DEBUG: Part {i}: {part[:100]}...", flush=True)
                if part.strip():
                    # Clean up any remaining prompt artifacts
                    cleaned = part.strip()
                    for prefix in ["assistant", "system", "user"]:
                        if cleaned.lower().startswith(prefix):
                            cleaned = cleaned[len(prefix):].strip()
                    cleaned_caption = clean_conversational_elements(cleaned)
                    captions.append(cleaned_caption)
                    print(f"   → DEBUG: Added caption: {cleaned_caption[:50]}...", flush=True)
                else:
                    print(f"   → DEBUG: Part {i} was empty", flush=True)
        
        # Ensure we have the right number of captions
        while len(captions) < len(frames):
            captions.append("No caption generated")
        
        print(f"   → generated {len(captions)} captions", flush=True)
        return captions[:len(frames)]  # Return only the number of frames we processed
        
    except Exception as e:
        print(f"Error in caption_frames_batch: {e}")
        traceback.print_exc()
        return ["Error captioning frame"] * len(frames)

def clean_conversational_elements(text):
    """Remove conversational elements from generated text."""
    if not text:
        return text
    
    # Remove common conversational phrases
    conversational_phrases = [
        "what do you think",
        "so what do you think",
        "well, ",
        "oh, ",
        "hmm, ",
        "okay, ",
        "so, ",
        "it seems like",
        "it looks like",
        "i think",
        "i believe",
        "in my opinion",
        "from what i can see",
        "as far as i can tell",
        "what do you think about",
        "what do you think of",
        "what do you think?",
        "what do you think about it?",
        "what do you think of it?",
        "so what do you think about",
        "well, it seems like",
        "well, it looks like",
        "well, i think",
        "well, from what i can see",
        "oh, it looks like",
        "oh, it seems like",
        "hmm, it looks like",
        "hmm, it seems like",
        "okay, it looks like",
        "okay, it seems like"
    ]
    
    cleaned_text = text.lower()
    for phrase in conversational_phrases:
        cleaned_text = cleaned_text.replace(phrase, "")
    
    # Remove question marks and related punctuation
    cleaned_text = cleaned_text.replace("?", "").replace("...", "").replace("..", "")
    
    # Capitalize first letter and clean up
    cleaned_text = cleaned_text.strip()
    if cleaned_text:
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
    
    return cleaned_text

def clear_output_file(output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared existing output file: {output_file}")

def load_processed_video_uids(output_file):
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

def find_video_path_from_clip_uid(clip_uid, clips_dir="../remote_ego4d/v2/clips"):
    """Find the video file path for a given clip UID."""
    video_path = os.path.join(clips_dir, f"{clip_uid}.mp4")
    if os.path.exists(video_path):
        return video_path
    else:
        print(f"Video file not found: {video_path}")
        return None

def process_nlq_videos_uniform(model, processor, output_file, max_videos=None, num_frames=32):
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
            print(f"Existing video UIDs: {[r['video_uid'] for r in existing_results]}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
            existing_results = []
    else:
        print(f"Output file {output_file} does not exist, starting fresh")
    
    existing_uids = {result['video_uid'] for result in existing_results}
    all_results = existing_results.copy()
    processed_count = 0
    
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
        video_path = find_video_path_from_clip_uid(clip_uid)
        if not video_path:
            print(f"Could not find video path for clip {clip_uid}, skipping...")
            continue
        
        try:
            # Extract frames uniformly from video
            frames = extract_uniform_frames_from_video(video_path, num_frames)
            if not frames:
                print(f"No frames extracted for video: {video_uid}")
                continue
            
            # Generate captions for frames in batches (much faster)
            frame_captions = []
            batch_size = 1  # Process 1 frame at a time for debugging
            
            for j in range(0, len(frames), batch_size):
                batch_frames = frames[j:j+batch_size]
                batch_captions = caption_frames_batch(model, processor, batch_frames)
                frame_captions.extend(batch_captions)
                print(f"Processed frames {j}-{j+len(batch_frames)-1}: {len(batch_captions)} captions")
            
            # Create video summary
            video_summary = summarize_captions(model, processor, frame_captions)
            
            # Create result entry (save both frame captions and summary for debugging)
            result = {
                'video_uid': video_uid,
                'clip_uid': clip_uid,
                'video_path': video_path,
                'original_summary': original_summary,
                'uniform_sampled_frames': len(frames),
                'frame_captions': frame_captions,  # Add frame captions for debugging
                'generated_summary': video_summary,
                'timestamp': datetime.now().isoformat()
            }
            
            all_results.append(result)
            
            # Save incrementally with better error handling
            try:
                print(f"   → Attempting to save {len(all_results)} results to {output_file}")
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"   → Successfully saved to file")
            except Exception as save_error:
                print(f"   → ERROR saving to file: {save_error}")
                traceback.print_exc()
                # Try to save to a backup file
                try:
                    backup_file = f"{output_file}.error_backup"
                    with open(backup_file, 'w') as f:
                        json.dump(all_results, f, indent=2)
                    print(f"   → Saved backup to {backup_file}")
                except Exception as backup_error:
                    print(f"   → ERROR saving backup: {backup_error}")
            
            processed_count += 1
            print(f"Completed and saved result for {video_uid}")
            
        except Exception as e:
            print(f"Error processing video {video_uid}: {e}")
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully processed {processed_count} new videos. Total results: {len(all_results)}")
    print(f"Results saved to {output_file}")
    return all_results

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ego4d_captioning.log'),
            logging.StreamHandler()
        ]
    )
    output_file = "ego4d_nlq_uniform_captions_qwen_omni.json"
    max_videos = None  # Process all videos (298 total)
    num_frames = 32
    print("Qwen2.5-Omni-7B Ego4D NLQ Validation Uniform Frame Captioning")
    print("=" * 50)
    print(f"NLQ validation videos from: ../v1/nlq_val_summaries.json")
    print(f"Video clips directory: ../remote_ego4d/v2/clips/")
    print(f"Output file: {output_file}")
    print(f"Max videos: {max_videos or 'All NLQ validation videos (298 total)'}")
    print(f"Frames per video: {num_frames}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    model, processor = load_model()
    if model is None or processor is None:
        print("Failed to load model. Exiting.")
        return
    
    # Check device mapping
    print(f"Model is on: {next(model.parameters()).device}")
    print(f"Processor device: {processor.device if hasattr(processor, 'device') else 'N/A'}")
    
    # Single-frame sanity check
    print("\n" + "="*50)
    print("Running single-frame sanity check...")
    print("="*50)
    
    # Test with a sample video
    test_video_path = "../remote_ego4d/v2/clips/000001.mp4"  # Adjust path as needed
    if os.path.exists(test_video_path):
        single = extract_uniform_frames_from_video(test_video_path, num_frames=1)
        if single:
            print("Single-frame test:", caption_frame(model, processor, single[0]))
        else:
            print("Could not extract test frame")
    else:
        print(f"Test video not found: {test_video_path}")
        print("Skipping single-frame test")
    
    print("="*50)
    print("Starting main processing...")
    print("="*50)
    
    process_nlq_videos_uniform(model, processor, output_file, max_videos, num_frames)
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main() 