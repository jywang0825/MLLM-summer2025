#!/usr/bin/env python3
"""
Prepare frame data for InternVL3 finetuning
- Extract 32 uniform frames from videos
- Use narration summaries for training/validation
- Create modified meta.json for frame-based training
"""

import json
import os
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from tqdm import tqdm
import random

def extract_uniform_frames(video_path: str, num_frames: int = 32) -> List[str]:
    """Extract uniformly distributed frames from a video."""
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print(f"Warning: Video has 0 frames: {video_path}")
            return []
        
        # Calculate frame indices for uniform sampling
        frame_indices = []
        for i in range(num_frames):
            frame_idx = int((i + 0.5) * total_frames / num_frames)
            frame_indices.append(frame_idx)
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Save frame to temporary file
                frame_filename = f"{Path(video_path).stem}_frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join("extracted_frames", frame_filename)
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
        
        cap.release()
        return frames
    
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def load_narration_summaries(summary_file: str) -> Dict[str, Dict[str, Any]]:
    """Load narration summaries from the processed results."""
    if not os.path.exists(summary_file):
        print(f"Warning: Summary file not found: {summary_file}")
        return {}
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create mapping of video_uid to summary data
    summaries = {}
    for item in data:
        if 'video_uid' in item:
            summaries[item['video_uid']] = item
    
    return summaries

def create_frame_meta(original_meta_path: str, 
                     narration_summaries: Dict[str, Dict[str, Any]],
                     output_dir: str = "internvl3_data_frames") -> str:
    """Create modified meta.json for frame-based training."""
    
    # Load original meta.json
    with open(original_meta_path, 'r', encoding='utf-8') as f:
        original_meta = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each video entry
    frame_meta = []
    
    for video_entry in tqdm(original_meta, desc="Processing videos"):
        video_uid = video_entry.get('video_uid', '')
        
        # Check if we have narration summary for this video
        if video_uid not in narration_summaries:
            print(f"Warning: No narration summary for video {video_uid}")
            continue
        
        summary_data = narration_summaries[video_uid]
        
        # Get video path from original meta
        video_path = video_entry.get('video_path', '')
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video path not found: {video_path}")
            continue
        
        # Extract 32 uniform frames
        frame_paths = extract_uniform_frames(video_path, num_frames=32)
        if not frame_paths:
            print(f"Warning: Could not extract frames from {video_path}")
            continue
        
        # Create frame-based training examples
        for i, frame_path in enumerate(frame_paths):
            # Use frame caption from summary data
            frame_caption = ""
            if 'frame_annotations' in summary_data and summary_data['frame_annotations']:
                # Get corresponding frame annotation
                if i < len(summary_data['frame_annotations']):
                    frame_caption = summary_data['frame_annotations'][i][1]  # (timestamp, caption)
                else:
                    # Fallback to summary if not enough frame annotations
                    frame_caption = summary_data.get('summary', 'Video frame')
            else:
                # Use overall summary if no frame annotations
                frame_caption = summary_data.get('summary', 'Video frame')
            
            # Create training example
            frame_example = {
                "id": f"{video_uid}_frame_{i:02d}",
                "video_uid": video_uid,
                "frame_path": frame_path,
                "caption": frame_caption,
                "summary": summary_data.get('summary', ''),
                "num_frames": len(frame_paths),
                "frame_index": i,
                "split": video_entry.get('split', 'train')
            }
            
            frame_meta.append(frame_example)
    
    # Save modified meta.json
    output_meta_path = os.path.join(output_dir, "meta_frames.json")
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(frame_meta, f, indent=2, ensure_ascii=False)
    
    print(f"Created frame-based meta.json with {len(frame_meta)} examples")
    print(f"Saved to: {output_meta_path}")
    
    return output_meta_path

def main():
    parser = argparse.ArgumentParser(description="Prepare frame data for InternVL3 finetuning")
    parser.add_argument("--original-meta", default="internvl3_data_all_videos/meta.json",
                       help="Path to original meta.json")
    parser.add_argument("--narration-summaries", 
                       default="narration_uniform_frame_summaries_all_train.json",
                       help="Path to narration summaries file")
    parser.add_argument("--output-dir", default="internvl3_data_frames",
                       help="Output directory for frame data")
    parser.add_argument("--num-frames", type=int, default=32,
                       help="Number of frames to extract per video")
    
    args = parser.parse_args()
    
    print("Loading narration summaries...")
    narration_summaries = load_narration_summaries(args.narration_summaries)
    print(f"Loaded {len(narration_summaries)} video summaries")
    
    print("Creating frame-based meta.json...")
    output_meta_path = create_frame_meta(
        args.original_meta,
        narration_summaries,
        args.output_dir
    )
    
    print(f"\nFrame data preparation completed!")
    print(f"Output meta.json: {output_meta_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Use --meta_path {output_meta_path} in your training script")

if __name__ == "__main__":
    main()



