#!/usr/bin/env python3
"""
Extract video frames from Ego4D dataset for InternVL3 LoRA finetuning
Optimized for video summarization training data preparation
"""

import cv2
import os
import argparse
from pathlib import Path
from typing import List, Optional
import json

def extract_frames_from_video(
    video_path: str, 
    output_dir: str, 
    fps: int = 1,
    max_frames: int = 10,
    quality: int = 95
) -> List[str]:
    """
    Extract frames from a video file at specified FPS
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        max_frames: Maximum number of frames to extract
        quality: JPEG quality (1-100)
    
    Returns:
        List of paths to extracted frame images
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if video_fps <= 0:
        print(f"Warning: Invalid FPS for {video_path}, using default 30")
        video_fps = 30
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    print(f"Video: {video_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Video FPS: {video_fps:.2f}")
    print(f"  Extract FPS: {fps}")
    print(f"  Frame interval: {frame_interval}")
    
    # Extract frames
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0 and extracted_count < max_frames:
            # Generate frame filename
            frame_filename = f"frame_{extracted_count+1:03d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    print(f"  Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths

def process_ego4d_dataset(
    ego4d_root: str,
    output_root: str = "video_frames",
    fps: int = 1,
    max_frames: int = 10,
    quality: int = 95,
    video_extensions: Optional[List[str]] = None
) -> dict:
    """
    Process Ego4D dataset to extract frames from all videos
    
    Args:
        ego4d_root: Root directory of Ego4D dataset
        output_root: Root directory for extracted frames
        fps: Frames per second to extract
        max_frames: Maximum frames per video
        quality: JPEG quality for saved frames
        video_extensions: List of video file extensions to process
    
    Returns:
        Dictionary with processing statistics
    """
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Find all video files
    video_files = []
    for root, dirs, files in os.walk(ego4d_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files in {ego4d_root}")
    
    # Process each video
    processed_videos = []
    failed_videos = []
    
    for i, video_path in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
        
        # Create output directory for this video
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_root, video_id)
        
        try:
            # Extract frames
            frame_paths = extract_frames_from_video(
                video_path, 
                video_output_dir, 
                fps, 
                max_frames, 
                quality
            )
            
            if frame_paths:
                processed_videos.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'frame_dir': video_output_dir,
                    'frame_paths': frame_paths,
                    'frame_count': len(frame_paths)
                })
            else:
                failed_videos.append(video_path)
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            failed_videos.append(video_path)
    
    # Save processing results
    results = {
        'total_videos': len(video_files),
        'processed_videos': len(processed_videos),
        'failed_videos': len(failed_videos),
        'extraction_settings': {
            'fps': fps,
            'max_frames': max_frames,
            'quality': quality
        },
        'processed_video_details': processed_videos
    }
    
    results_path = os.path.join(output_root, 'frame_extraction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*50)
    print("Frame Extraction Complete!")
    print("="*50)
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {len(processed_videos)}")
    print(f"Failed: {len(failed_videos)}")
    print(f"Results saved to: {results_path}")
    print(f"Frames saved to: {output_root}")
    print("\n🚀 Ready for InternVL3 LoRA finetuning!")
    print("Next step: Run prepare_internvl3_finetuning.py")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract video frames for InternVL3 LoRA finetuning")
    parser.add_argument("--ego4d_root", type=str, required=True,
                       help="Root directory of Ego4D dataset")
    parser.add_argument("--output_root", type=str, default="video_frames",
                       help="Output directory for extracted frames")
    parser.add_argument("--fps", type=int, default=1,
                       help="Frames per second to extract (default: 1)")
    parser.add_argument("--max_frames", type=int, default=10,
                       help="Maximum frames per video (default: 10)")
    parser.add_argument("--quality", type=int, default=95,
                       help="JPEG quality (1-100, default: 95)")
    parser.add_argument("--video_extensions", nargs="+", 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                       help="Video file extensions to process")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.ego4d_root):
        print(f"Error: Ego4D root directory does not exist: {args.ego4d_root}")
        return
    
    if args.fps <= 0:
        print("Error: FPS must be positive")
        return
    
    if args.max_frames <= 0:
        print("Error: Max frames must be positive")
        return
    
    if not (1 <= args.quality <= 100):
        print("Error: Quality must be between 1 and 100")
        return
    
    # Process dataset
    process_ego4d_dataset(
        args.ego4d_root,
        args.output_root,
        args.fps,
        args.max_frames,
        args.quality,
        args.video_extensions
    )

if __name__ == "__main__":
    main() 