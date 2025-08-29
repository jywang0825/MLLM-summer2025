#!/usr/bin/env python3
"""
Rewritten AKS for Ego4D NLQ Dataset - Uses "Describe the main activity in this video" prompt
Generates descriptions from frames instead of matching frames to summaries
"""

import torch
import json
import numpy as np
import os
import argparse
import heapq
from PIL import Image
from decord import VideoReader, cpu
import cv2
try:
    from lavis.models import load_model_and_preprocess
except ImportError:
    print("Warning: LAVIS not found. Please install it for BLIP support.")
    load_model_and_preprocess = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    try:
        from transformers.models.clip.processing_clip import CLIPProcessor
        from transformers.models.clip.modeling_clip import CLIPModel
    except ImportError:
        print("Warning: CLIP not found. Please install transformers.")
        CLIPProcessor = None
        CLIPModel = None
from tqdm import tqdm

def load_manifest(manifest_path):
    """Load clips manifest to map parent_video_uid to exported_clip_uid and filename using manifold_location."""
    import pandas as pd
    
    df = pd.read_csv(manifest_path)
    parent_to_clips = {}
    
    for _, row in df.iterrows():
        parent_video_uid = str(row['parent_video_uid'])
        exported_clip_uid = str(row['exported_clip_uid'])
        manifold_location = str(row['manifold_location'])
        
        local_path = manifold_location.replace('manifold://ego4d_fair/tree/exported_clips/', '../remote_ego4d/v2/clips/')
        
        if parent_video_uid not in parent_to_clips:
            parent_to_clips[parent_video_uid] = []
        parent_to_clips[parent_video_uid].append({
            'exported_clip_uid': exported_clip_uid,
            'local_path': local_path,
            'filename': os.path.basename(local_path)
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
        if CLIPModel is None or CLIPProcessor is None:
            raise ValueError("CLIP not installed. Please install transformers.")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor, None
    else:
        raise ValueError(f"Model {model_name} not supported")

def extract_features_for_video(video_path, model, vis_processors, text_processors, 
                              model_name, device):
    """Extract features for a single video using 'Describe the main activity' prompt."""
    
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
        
        # Use the new prompt: "Describe the main activity in this video"
        query_text = "Describe the main activity in this video"
        
        print(f"Processing {min(frame_nums, 300)} frames with prompt: '{query_text}'")
        
        # Process all frames (up to 300 for efficiency)
        for j in range(min(frame_nums, 300)):
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

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    """AKS algorithm for adaptive frame selection."""
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    i = 0
    
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = np.mean(score)
        std = np.std(score)

        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        i += 1
        
        mean_diff = np.mean(top_score) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score)//2]
            score2 = score[len(score)//2:]
            fn1 = fn[:len(score)//2]
            fn2 = fn[len(score)//2:]
            split_scores.append(dict(score=score1, depth=depth+1))
            split_scores.append(dict(score=score2, depth=depth+1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    
    if len(split_scores) > 0:
        all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        all_split_score = []
        all_split_fn = []
    
    all_split_score = no_split_scores + all_split_score
    all_split_fn = no_split_fn + all_split_fn

    return all_split_score, all_split_fn

def select_frames_aks(scores, frame_indices, max_num_frames, t1, t2, all_depth):
    """Apply AKS algorithm to select frames."""
    if len(scores) <= max_num_frames:
        return frame_indices
    
    # Normalize scores
    scores = np.array(scores)
    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    # Apply AKS algorithm
    a, b = meanstd(len(scores), [dict(score=normalized_scores, depth=0)], 
                   max_num_frames, [frame_indices], t1, t2, all_depth)
    
    selected_frames = []
    for s, f in zip(a, b):
        f_num = int(max_num_frames / 2**(s['depth']))
        topk = heapq.nlargest(f_num, range(len(s['score'])), s['score'].__getitem__)
        f_nums = [f[t] for t in topk]
        selected_frames.extend(f_nums)
    
    selected_frames.sort()
    return selected_frames

def extract_and_save_frames(video_path, selected_frames, output_dir, video_uid):
    """Extract and save selected frames as images."""
    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        
        # Create video-specific directory
        video_dir = os.path.join(output_dir, 'frames', video_uid)
        os.makedirs(video_dir, exist_ok=True)
        
        saved_frames = []
        for i, frame_idx in enumerate(selected_frames):
            try:
                frame = np.array(vr[frame_idx])
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Save frame
                frame_filename = f"frame_{i:03d}_{frame_idx:06d}.jpg"
                frame_path = os.path.join(video_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                saved_frames.append({
                    'frame_index': frame_idx,
                    'frame_filename': frame_filename,
                    'frame_path': frame_path
                })
                
            except Exception as e:
                print(f"Error saving frame {frame_idx}: {e}")
                continue
        
        return saved_frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def save_progress(results, output_file):
    """Save current progress to file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def load_progress(output_file):
    """Load existing progress from file."""
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return []

def load_nlq_val_videos(summaries_path="./nlq_val_summaries.json", nlq_val_path="../remote_ego4d/v2/annotations/nlq_val.json"):
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

def find_video_path(clip_uid, clips_manifest_path="../remote_ego4d/v2/clips/manifest.csv"):
    """Find the video file path for a given clip UID from the manifest."""
    try:
        import pandas as pd
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

def main():
    # Parameters
    nlq_val_path = './nlq_val_summaries.json'
    extract_feature_model = 'clip'  # or 'blip'
    output_dir = './ego4d_aks_rewritten'
    max_num_frames = 32
    device = 'cpu'  # Use CPU to avoid GPU memory issues
    t1 = 0.8
    t2 = -100
    all_depth = 5
    
    print("🎬 Rewritten AKS Frame Selector")
    print("📝 Using prompt: 'Describe the main activity in this video'")
    print(f"🚀 Device: {device}")
    print(f"📦 Output directory: {output_dir}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
    
    # Load NLQ validation data using the same approach as uniform script
    video_data = load_nlq_val_videos()
    
    print(f"Processing {len(video_data)} videos from NLQ validation set")
    
    # Load model
    print(f"Loading {extract_feature_model} model...")
    model, vis_processors, text_processors = load_model(extract_feature_model, device)
    
    # Load existing progress
    output_file = os.path.join(output_dir, 'ego4d_aks_rewritten_results.json')
    results = load_progress(output_file)
    processed_uids = {r['video_uid'] for r in results}
    
    print(f"Found {len(results)} already processed videos")
    
    # Process each video
    for i, video_info in enumerate(tqdm(video_data, desc="Processing videos")):
        video_uid = video_info['video_uid']
        clip_uid = video_info['clip_uid']
        original_summary = video_info['original_summary']
        has_clips = video_info['has_clips']
        
        # Skip if already processed
        if video_uid in processed_uids:
            continue
        
        if not has_clips:
            print(f"Video {video_uid} has no clips, skipping...")
            continue
        
        print(f"\n📹 Processing video {i+1}/{len(video_data)}: {video_uid}")
        print(f"🎬 Clip UID: {clip_uid}")
        print(f"📝 Original summary: {original_summary}")
        
        # Find video file path using clip_uid
        video_path_full = find_video_path(clip_uid)
        if not video_path_full:
            print(f"Could not find video path for clip {clip_uid}, skipping...")
            continue
        
        # Extract features using the new prompt
        scores, frame_indices = extract_features_for_video(
            video_path_full, model, vis_processors, text_processors,
            extract_feature_model, device
        )
        
        if len(scores) == 0:
            print(f"No features extracted for {video_uid}")
            continue
        
        # Apply AKS frame selection
        selected_frames = select_frames_aks(
            scores, frame_indices, max_num_frames, t1, t2, all_depth
        )
        
        # Extract and save frames
        saved_frames = extract_and_save_frames(
            video_path_full, selected_frames, output_dir, video_uid
        )
        
        # Store results
        result = {
            'video_uid': video_uid,
            'clip_uid': clip_uid,
            'original_summary': original_summary,
            'video_path': video_path_full,
            'prompt_used': "Describe the main activity in this video",
            'num_frames_processed': len(scores),
            'all_scores': scores,
            'all_frames': frame_indices,
            'selected_frames': selected_frames,
            'saved_frames': saved_frames,
            'num_selected': len(selected_frames)
        }
        results.append(result)
        processed_uids.add(video_uid)
        
        # Save progress after each video
        save_progress(results, output_file)
        
        print(f"✅ Selected {len(selected_frames)} frames, saved {len(saved_frames)} images")
    
    # Final statistics
    print(f"\n🎉 AKS processing completed!")
    print(f"📊 Processed {len(results)} videos")
    print(f"💾 Results saved to {output_file}")
    
    # Print statistics
    num_frames = [r['num_selected'] for r in results]
    print(f"📈 Average frames selected: {np.mean(num_frames):.2f}")
    print(f"📉 Min frames selected: {np.min(num_frames)}")
    print(f"📈 Max frames selected: {np.max(num_frames)}")

if __name__ == '__main__':
    main() 