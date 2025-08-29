#!/usr/bin/env python3
"""
Enhanced Ego4D preparation for InternVL3 finetuning with frame-by-frame annotations.
Supports both video summaries and detailed frame narrations for richer training.
"""

import json
import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime

class EnhancedEgo4DProcessor:
    def __init__(self, ego4d_root: str, output_dir: str):
        self.ego4d_root = Path(ego4d_root)
        self.output_dir = Path(output_dir)
        self.video_frames_dir = self.output_dir / "video_frames"
        self.video_frames_dir.mkdir(parents=True, exist_ok=True)
        
    def load_video_summaries(self, summaries_file: str) -> Dict:
        """Load video summaries from JSON file."""
        summaries_path = self.ego4d_root / "finetuning" / "nlq_summaries" / summaries_file
        print(f"Loading video summaries from: {summaries_path}")
        
        with open(summaries_path, 'r') as f:
            return json.load(f)
    
    def find_video_file(self, video_uid: str) -> Optional[Path]:
        """Find video file by UID in various possible locations."""
        possible_paths = [
            self.ego4d_root / "v2" / "clips" / f"{video_uid}.mp4",
            self.ego4d_root / "v1" / "clips" / f"{video_uid}.mp4",
            self.ego4d_root / "clips" / f"{video_uid}.mp4",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def extract_frame_at_timestamp(self, video_path: Path, video_uid: str, 
                                 timestamp_sec: float, frame_name: str) -> Optional[str]:
        """Extract a single frame at specific timestamp."""
        output_dir = self.video_frames_dir / video_uid
        output_dir.mkdir(exist_ok=True)
        
        frame_path = output_dir / f"{frame_name}.jpg"
        
        # Extract frame at timestamp
        cmd = [
            "ffmpeg", "-i", str(video_path), "-ss", str(timestamp_sec),
            "-vframes", "1", "-q:v", "2", str(frame_path), "-y"
        ]
        subprocess.run(cmd, capture_output=True)
        
        if frame_path.exists():
            return f"video_frames/{video_uid}/{frame_name}.jpg"
        return None
    
    def create_summary_training_data(self, summaries: Dict, max_videos: Optional[int] = None) -> Tuple[List, List]:
        """Create training data using video summaries (current approach)."""
        train_data = []
        val_data = []
        
        video_items = list(summaries.items())
        if max_videos:
            video_items = video_items[:max_videos]
        
        # Split into train/val (80/20)
        random.shuffle(video_items)
        split_idx = int(len(video_items) * 0.8)
        train_items = video_items[:split_idx]
        val_items = video_items[split_idx:]
        
        print(f"Creating summary training data: {len(train_items)} train, {len(val_items)} val")
        
        for split_name, items in [("train", train_items), ("val", val_items)]:
            for video_uid, video_data in items:
                # Find video file
                video_path = self.find_video_file(video_uid)
                if not video_path:
                    continue
                
                # Get summary
                summary = video_data.get("summary", "")
                if not summary:
                    continue
                
                # Extract 4 frames uniformly
                frame_paths = self.extract_frames_uniform(video_path, video_uid, 4)
                if not frame_paths:
                    continue
                
                # Create training example
                example = {
                    "id": f"{split_name}_summary_{video_uid}",
                    "image": frame_paths[0],  # Use first frame for summary
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nPlease provide a concise summary of this video."
                        },
                        {
                            "from": "assistant",
                            "value": summary.replace("#Summary ", "").strip()
                        }
                    ]
                }
                
                if split_name == "train":
                    train_data.append(example)
                else:
                    val_data.append(example)
        
        return train_data, val_data
    
    def create_narration_training_data(self, summaries: Dict, max_videos: Optional[int] = None) -> Tuple[List, List]:
        """Create training data using frame-by-frame narrations."""
        train_data = []
        val_data = []
        
        video_items = list(summaries.items())
        if max_videos:
            video_items = video_items[:max_videos]
        
        # Split into train/val (80/20)
        random.shuffle(video_items)
        split_idx = int(len(video_items) * 0.8)
        train_items = video_items[:split_idx]
        val_items = video_items[split_idx:]
        
        print(f"Creating narration training data: {len(train_items)} train, {len(val_items)} val")
        
        for split_name, items in [("train", train_items), ("val", val_items)]:
            for video_uid, video_data in items:
                # Find video file
                video_path = self.find_video_file(video_uid)
                if not video_path:
                    continue
                
                # Get narrations
                narrations = []
                if "video_data" in video_data and "narration_pass_1" in video_data["video_data"]:
                    narrations = video_data["video_data"]["narration_pass_1"].get("narrations", [])
                
                if not narrations:
                    continue
                
                # Create training example for each narration
                for i, narration in enumerate(narrations):
                    if "timestamp_sec" not in narration or "narration_text" not in narration:
                        continue
                    
                    # Extract frame at narration timestamp
                    frame_path = self.extract_frame_at_timestamp(
                        video_path, video_uid, 
                        narration["timestamp_sec"], 
                        f"narration_{i:03d}"
                    )
                    
                    if not frame_path:
                        continue
                    
                    # Clean narration text
                    narration_text = narration["narration_text"]
                    # Remove Ego4D-specific tags
                    narration_text = narration_text.replace("#C ", "").replace("#unsure", "").strip()
                    
                    if not narration_text:
                        continue
                    
                    # Create training example
                    example = {
                        "id": f"{split_name}_narration_{video_uid}_{i}",
                        "image": frame_path,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\nWhat is happening in this moment?"
                            },
                            {
                                "from": "assistant",
                                "value": narration_text
                            }
                        ]
                    }
                    
                    if split_name == "train":
                        train_data.append(example)
                    else:
                        val_data.append(example)
        
        return train_data, val_data
    
    def extract_frames_uniform(self, video_path: Path, video_uid: str, num_frames: int = 4) -> List[str]:
        """Extract frames uniformly distributed across the video."""
        output_dir = self.video_frames_dir / video_uid
        output_dir.mkdir(exist_ok=True)
        
        # Get video duration
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        # Calculate frame timestamps
        timestamps = [duration * i / (num_frames - 1) for i in range(num_frames)]
        if num_frames == 1:
            timestamps = [duration / 2]
        
        frame_paths = []
        for i, timestamp in enumerate(timestamps):
            frame_path = output_dir / f"frame_{i:03d}.jpg"
            
            # Extract frame at timestamp
            cmd = [
                "ffmpeg", "-i", str(video_path), "-ss", str(timestamp),
                "-vframes", "1", "-q:v", "2", str(frame_path), "-y"
            ]
            subprocess.run(cmd, capture_output=True)
            
            if frame_path.exists():
                frame_paths.append(f"video_frames/{video_uid}/frame_{i:03d}.jpg")
        
        return frame_paths
    
    def create_hybrid_training_data(self, summaries: Dict, max_videos: Optional[int] = None) -> Tuple[List, List]:
        """Create hybrid training data using both summaries and narrations."""
        print("Creating hybrid training data (summaries + narrations)...")
        
        # Get both types of data
        summary_train, summary_val = self.create_summary_training_data(summaries, max_videos)
        narration_train, narration_val = self.create_narration_training_data(summaries, max_videos)
        
        # Combine the data
        train_data = summary_train + narration_train
        val_data = summary_val + narration_val
        
        # Shuffle the combined data
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        print(f"Hybrid dataset created:")
        print(f"  Summary examples: {len(summary_train)} train, {len(summary_val)} val")
        print(f"  Narration examples: {len(narration_train)} train, {len(narration_val)} val")
        print(f"  Total: {len(train_data)} train, {len(val_data)} val")
        
        return train_data, val_data
    
    def save_training_data(self, train_data: List, val_data: List, data_type: str = "hybrid"):
        """Save training data to JSONL files."""
        # Save training data
        train_file = self.output_dir / f"video_summary_dataset_train.jsonl"
        with open(train_file, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
        
        # Save validation data
        val_file = self.output_dir / f"video_summary_dataset_val.jsonl"
        with open(val_file, 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        # Create meta.json
        meta = {
            "video_summary_dataset": {
                "root": "video_frames/",
                "annotation": "internvl3_data/video_summary_dataset_train.jsonl",
                "data_augment": False,
                "max_dynamic_patch": 12,
                "repeat_time": 1,
                "length": len(train_data[0]["conversations"]) if train_data else 0
            }
        }
        
        meta_file = self.output_dir / "meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\nSaved {len(train_data)} training examples to {train_file}")
        print(f"Saved {len(val_data)} validation examples to {val_file}")
        print(f"Saved meta.json to {meta_file}")
        print(f"Data type: {data_type}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Ego4D preparation for InternVL3 finetuning")
    parser.add_argument("--ego4d-root", default="../remote_ego4d", 
                       help="Path to Ego4D dataset root")
    parser.add_argument("--output-dir", default="internvl3_data", 
                       help="Output directory for processed data")
    parser.add_argument("--summaries-file", default="nlq_video_summaries.json",
                       choices=["nlq_video_summaries.json", "nlq_summaries_dataset.json"],
                       help="Video summaries file to use")
    parser.add_argument("--data-type", default="hybrid",
                       choices=["summary", "narration", "hybrid"],
                       help="Type of training data to create")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = EnhancedEgo4DProcessor(args.ego4d_root, args.output_dir)
    
    # Load video summaries
    summaries = processor.load_video_summaries(args.summaries_file)
    
    # Create training data based on type
    if args.data_type == "summary":
        train_data, val_data = processor.create_summary_training_data(summaries, args.max_videos)
    elif args.data_type == "narration":
        train_data, val_data = processor.create_narration_training_data(summaries, args.max_videos)
    else:  # hybrid
        train_data, val_data = processor.create_hybrid_training_data(summaries, args.max_videos)
    
    # Save training data
    processor.save_training_data(train_data, val_data, args.data_type)
    
    print(f"\nProcessing complete!")
    print(f"Data type: {args.data_type}")
    print(f"Total training examples: {len(train_data)}")
    print(f"Total validation examples: {len(val_data)}")

if __name__ == "__main__":
    main() 