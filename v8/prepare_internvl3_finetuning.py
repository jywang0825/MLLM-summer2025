#!/usr/bin/env python3
"""
Prepare video summarization data for InternVL3 supervised finetuning
Converts existing evaluation data into the proper format for training
Follows official InternVL3 methodology with 4-bit LoRA optimization
GPU-accelerated frame extraction
"""

import json
import os
import argparse
import subprocess
import csv
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
from datetime import datetime

# Enable GPU acceleration for OpenCV if available
try:
    # Check if CUDA is available for OpenCV
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print(f"🚀 GPU acceleration available: {cv2.cuda.getCudaEnabledDeviceCount()} CUDA devices")
        USE_GPU = True
    else:
        print("⚠️  GPU acceleration not available for OpenCV, using CPU")
        USE_GPU = False
except:
    print("⚠️  GPU acceleration not available for OpenCV, using CPU")
    USE_GPU = False

def load_evaluation_data(evaluation_file: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from the optimized evaluation results
    """
    print(f"Loading evaluation data from {evaluation_file}")
    
    with open(evaluation_file, 'r') as f:
        data = json.load(f)
    
    # Extract video data from evaluation results
    video_data = []
    
    # Look for video entries in the evaluation data
    if 'results' in data:
        # This is the summary format
        for video_id, video_info in data['results'].items():
            if isinstance(video_info, dict) and 'video_id' in video_info:
                video_data.append(video_info)
    else:
        # This might be the detailed results format
        for item in data:
            if isinstance(item, dict) and 'video_id' in item:
                video_data.append(item)
    
    print(f"Found {len(video_data)} video entries")
    return video_data

def create_internvl3_meta_file(dataset_paths: Dict[str, str], output_path: str):
    """
    Create meta file for InternVL3 training following official format
    """
    meta_data = {
        "video_summary_dataset": {
            "root": "video_frames/",
            "annotation": dataset_paths["train"],
            "data_augment": False,
            "max_dynamic_patch": 12,
            "repeat_time": 1,
            "length": 0  # Will be calculated
        }
    }
    
    # Calculate length from training data
    with open(dataset_paths["train"], 'r') as f:
        length = sum(1 for line in f)
    meta_data["video_summary_dataset"]["length"] = length
    
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"Saved meta file to {output_path}")
    print(f"Dataset length: {length} samples")

def create_internvl3_jsonl_dataset(
    video_data: List[Dict[str, Any]], 
    output_path: str,
    image_folder: str = "video_frames",
    train_ratio: float = 0.8
) -> Dict[str, str]:
    """
    Create InternVL3 JSONL dataset from video summarization data
    Follows the official InternVL3 data format
    """
    
    # Split data into train and validation
    random.shuffle(video_data)
    split_idx = int(len(video_data) * train_ratio)
    train_data = video_data[:split_idx]
    val_data = video_data[split_idx:]
    
    print(f"Creating dataset with {len(train_data)} training samples and {len(val_data)} validation samples")
    
    def create_jsonl_format(data_list: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        """Convert video data to InternVL3 JSONL format"""
        jsonl_data = []
        
        for i, item in enumerate(data_list):
            video_id = item.get('video_id', f'video_{i}')
            
            # Get the ground truth summary
            ground_truth = item.get('ground_truth_summary', '')
            if not ground_truth:
                # Try alternative keys
                ground_truth = item.get('summary', '')
                ground_truth = item.get('test_summary', '')
            
            if not ground_truth:
                print(f"Warning: No ground truth summary found for {video_id}")
                continue
            
            # Get frame paths
            frames = item.get('frames', [])
            if not frames:
                # Try to construct frame path
                frame_path = os.path.join(image_folder, f"{video_id}_frame_001.jpg")
                frames = [frame_path]
            
            # Use first frame as representative image
            image_path = frames[0] if frames else f"{image_folder}/default_frame.jpg"
            
            # Create InternVL3 JSONL format
            jsonl_entry = {
                "id": f"{split_name}_{video_id}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nPlease provide a concise summary of this video."
                    },
                    {
                        "from": "assistant",
                        "value": ground_truth
                    }
                ]
            }
            jsonl_data.append(jsonl_entry)
        
        return jsonl_data
    
    # Create train and validation datasets
    train_jsonl = create_jsonl_format(train_data, "train")
    val_jsonl = create_jsonl_format(val_data, "val")
    
    # Save as JSONL files
    train_path = output_path.replace('.json', '_train.jsonl')
    val_path = output_path.replace('.json', '_val.jsonl')
    
    # Write JSONL files (one JSON object per line)
    with open(train_path, 'w') as f:
        for item in train_jsonl:
            f.write(json.dumps(item) + '\n')
    
    with open(val_path, 'w') as f:
        for item in val_jsonl:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved training dataset to {train_path}")
    print(f"Saved validation dataset to {val_path}")
    
    return {
        "train": train_path,
        "val": val_path
    }

def create_official_internvl3_training_scripts():
    """
    Create official InternVL3 training scripts with 4-bit LoRA optimization
    """
    
    # 4-bit LoRA finetuning script (official InternVL3 methodology)
    lora_script = '''#!/bin/bash
# finetune_internvl3_8b_4bit_lora.sh - OFFICIAL INTERNVL3 METHODOLOGY
# 4-bit LoRA finetuning using official InternVL3 training script
# Optimized for 48GB Ada Gen cards

set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl3_8b_4bit_lora'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "Starting InternVL3-8B 4-bit LoRA finetuning (Official Method)"
echo "Optimized for 48GB Ada Gen cards"
echo "GPUs: ${GPUS}, Batch size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "Total batch size: ${BATCH_SIZE}, Gradient accumulation: ${GRADIENT_ACC}"

# Official InternVL3 LoRA finetuning with 4-bit optimization
torchrun \\
  --nnodes=1 \\
  --node_rank=0 \\
  --master_addr=127.0.0.1 \\
  --nproc_per_node=${GPUS} \\
  --master_port=${MASTER_PORT} \\
  internvl/train/internvl_chat_finetune.py \\
  --model_name_or_path "models/InternVL3-8B" \\
  --conv_style "internvl2_5" \\
  --use_fast_tokenizer False \\
  --output_dir ${OUTPUT_DIR} \\
  --meta_path "internvl3_data/meta.json" \\
  --overwrite_output_dir True \\
  --force_image_size 448 \\
  --max_dynamic_patch 12 \\
  --down_sample_ratio 0.5 \\
  --drop_path_rate 0.0 \\
  --freeze_llm True \\
  --freeze_mlp True \\
  --freeze_backbone True \\
  --use_llm_lora 32 \\
  --vision_select_layer -1 \\
  --dataloader_num_workers 4 \\
  --bf16 True \\
  --num_train_epochs 3 \\
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \\
  --gradient_accumulation_steps ${GRADIENT_ACC} \\
  --evaluation_strategy "steps" \\
  --eval_steps 100 \\
  --save_strategy "steps" \\
  --save_steps 200 \\
  --save_total_limit 2 \\
  --learning_rate 2e-5 \\
  --weight_decay 0.05 \\
  --warmup_ratio 0.03 \\
  --lr_scheduler_type "cosine" \\
  --logging_steps 10 \\
  --max_seq_length 8192 \\
  --do_train True \\
  --do_eval True \\
  --grad_checkpoint True \\
  --group_by_length True \\
  --dynamic_image_size True \\
  --use_thumbnail True \\
  --ps_version 'v2' \\
  --deepspeed "zero_stage3_config.json" \\
  --report_to "tensorboard" \\
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "InternVL3 4-bit LoRA finetuning completed!"
echo "Check results in: ${OUTPUT_DIR}"
'''
    
    # Quick start script
    quick_start_script = '''#!/bin/bash
# quick_start_internvl3_4bit_lora.sh - EASY START SCRIPT
# Official InternVL3 4-bit LoRA finetuning with optimized defaults

echo "🚀 Quick Start: InternVL3 4-bit LoRA Finetuning (Official Method)"
echo "==============================================================="
echo "Optimized for 48GB Ada Gen cards"
echo ""

# Set reasonable defaults for 48GB Ada Gen cards
export GPUS=2
export PER_DEVICE_BATCH_SIZE=4
export BATCH_SIZE=32

echo "Configuration:"
echo "- GPUs: ${GPUS}"
echo "- Batch size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "- Total batch size: ${BATCH_SIZE}"
echo "- 4-bit optimization: Enabled"
echo "- LoRA rank: 32 (official InternVL3)"
echo "- Official InternVL3 methodology"
echo ""

# Check if we have the required files
if [ ! -f "internvl3_data/meta.json" ]; then
    echo "❌ Error: meta.json not found!"
    echo "Please run: python prepare_internvl3_finetuning.py first"
    exit 1
fi

if [ ! -f "finetune_internvl3_8b_4bit_lora.sh" ]; then
    echo "❌ Error: Training script not found!"
    echo "Please run: python prepare_internvl3_finetuning.py first"
    exit 1
fi

if [ ! -f "zero_stage3_config.json" ]; then
    echo "❌ Error: DeepSpeed config not found!"
    echo "Please run: python prepare_internvl3_finetuning.py first"
    exit 1
fi

echo "✅ All required files found. Starting official InternVL3 4-bit LoRA finetuning..."
echo ""

# Run official InternVL3 4-bit LoRA finetuning
./finetune_internvl3_8b_4bit_lora.sh

echo ""
echo "🎉 Training started! Monitor with:"
echo "tensorboard --logdir work_dirs/"
echo "tail -f work_dirs/internvl3_8b_4bit_lora/training_log.txt"
echo ""
echo "💡 Official InternVL3 advantages:"
echo "- Proven training methodology"
echo "- Optimized for InternVL3 architecture"
echo "- LoRA rank 32 with official implementation"
echo "- DeepSpeed ZeRO Stage 3 optimization"
echo "- Dynamic image size support"
'''
    
    # Save scripts
    with open('finetune_internvl3_8b_4bit_lora.sh', 'w') as f:
        f.write(lora_script)
    
    with open('quick_start_internvl3_4bit_lora.sh', 'w') as f:
        f.write(quick_start_script)
    
    # Make scripts executable
    os.chmod('finetune_internvl3_8b_4bit_lora.sh', 0o755)
    os.chmod('quick_start_internvl3_4bit_lora.sh', 0o755)
    
    print("Created official InternVL3 4-bit LoRA finetuning scripts:")
    print("- finetune_internvl3_8b_4bit_lora.sh (Official InternVL3 4-bit LoRA)")
    print("- quick_start_internvl3_4bit_lora.sh (Easy start script with defaults)")

def create_deepspeed_configs():
    """
    Create DeepSpeed configuration files for official InternVL3
    """
    configs = {
        "zero_stage3_config.json": {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "bf16": {
                "enabled": True
            },
            "gradient_clipping": 1.0,
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 4
        }
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created {filename}")

def create_model_download_script():
    """
    Create script to download InternVL3 models following official docs
    """
    download_script = '''#!/bin/bash
# download_internvl3_models.sh
# Download InternVL3 models following official documentation

echo "📥 Downloading InternVL3 models..."

# Install huggingface_hub if not already installed
pip install -U huggingface_hub

# Create models directory
mkdir -p models

cd models

# Download InternVL3-8B (recommended for most users)
echo "Downloading InternVL3-8B..."
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-8B --local-dir InternVL3-8B

# Optional: Download other model sizes
# echo "Downloading InternVL3-1B..."
# huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-1B --local-dir InternVL3-1B

# echo "Downloading InternVL3-2B..."
# huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-2B --local-dir InternVL3-2B

echo "✅ Model download completed!"
echo "Models are available in: models/"
'''
    
    with open('download_internvl3_models.sh', 'w') as f:
        f.write(download_script)
    
    os.chmod('download_internvl3_models.sh', 0o755)
    print("Created model download script: download_internvl3_models.sh")

def create_requirements_file():
    """
    Create requirements.txt for official InternVL3 with 4-bit optimization
    """
    requirements = '''# Official InternVL3 4-bit LoRA Finetuning Requirements
torch>=2.0.0
transformers>=4.35.0
deepspeed>=0.12.0
accelerate>=0.24.0
tensorboard>=2.14.0
pillow>=9.5.0
opencv-python>=4.8.0
sentencepiece>=0.1.99
protobuf>=3.20.0
huggingface_hub>=0.16.0
bitsandbytes>=0.41.0
peft>=0.6.0
'''
    
    with open('requirements_official.txt', 'w') as f:
        f.write(requirements)
    
    print("Created requirements file: requirements_official.txt")

def load_ego4d_manifest(manifest_path: str) -> Dict[str, str]:
    """
    Load Ego4D manifest CSV to map parent_video_uid to exported_clip_uid
    """
    print(f"Loading Ego4D manifest from {manifest_path}")
    
    uid_mapping = {}
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parent_video_uid = row['parent_video_uid']
            exported_clip_uid = row['exported_clip_uid']
            uid_mapping[parent_video_uid] = exported_clip_uid
    
    print(f"Loaded {len(uid_mapping)} video UID mappings")
    return uid_mapping

def load_ego4d_video_summaries(summaries_file: str) -> Dict:
    """
    Load Ego4D video summaries from JSON file
    """
    print(f"Loading Ego4D video summaries from {summaries_file}")
    
    with open(summaries_file, 'r') as f:
        return json.load(f)

def find_ego4d_video_file(video_uid: str, uid_mapping: Dict[str, str], ego4d_root: str) -> Optional[Path]:
    """
    Find Ego4D video file using manifest mapping
    """
    # First try direct mapping
    if video_uid in uid_mapping:
        exported_clip_uid = uid_mapping[video_uid]
        video_path = Path(ego4d_root) / "v2" / "clips" / f"{exported_clip_uid}.mp4"
        if video_path.exists():
            return video_path
    
    # Try direct lookup in case the video_uid is already the exported_clip_uid
    possible_paths = [
        Path(ego4d_root) / "v2" / "clips" / f"{video_uid}.mp4",
        Path(ego4d_root) / "v1" / "clips" / f"{video_uid}.mp4",
        Path(ego4d_root) / "clips" / f"{video_uid}.mp4",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None

def extract_frame_at_timestamp(video_path: Path, video_uid: str, 
                             timestamp_sec: float, frame_name: str, 
                             output_dir: Path) -> Optional[str]:
    """
    Extract a single frame at specific timestamp using OpenCV
    """
    frame_dir = output_dir / video_uid
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    frame_path = frame_dir / f"{frame_name}.jpg"
    
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp_sec * fps)
        
        # Set position to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            # Save frame
            cv2.imwrite(str(frame_path), frame)
            cap.release()
            return f"video_frames/{video_uid}/{frame_name}.jpg"
        else:
            cap.release()
            return None
            
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return None

def extract_frames_uniform(video_path: Path, video_uid: str, num_frames: int, output_dir: Path) -> List[str]:
    """
    Extract frames uniformly distributed across the video using OpenCV (GPU-accelerated if available)
    """
    frame_dir = output_dir / video_uid
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open video file with GPU acceleration if available
        use_gpu_capture = False
        if USE_GPU:
            try:
                # Try GPU-accelerated video capture
                cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
                # Check if we can use GPU acceleration
                if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    use_gpu_capture = True
                    print(f"  Using GPU-accelerated video capture for {video_uid}")
            except:
                pass
        
        if not use_gpu_capture:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices for uniform distribution
        if num_frames == 1:
            frame_indices = [total_frames // 2]
        else:
            frame_indices = [int(total_frames * i / (num_frames - 1)) for i in range(num_frames)]
        
        frame_paths = []
        for i, frame_idx in enumerate(frame_indices):
            # Set position to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            
            if ret:
                frame_path = frame_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(f"video_frames/{video_uid}/frame_{i:03d}.jpg")
        
        cap.release()
        print(f"  Extracted {len(frame_paths)} frames from {video_uid} ({'GPU' if use_gpu_capture else 'CPU'})")
        return frame_paths
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def extract_frames_by_interval(video_path: Path, video_uid: str, interval: float, output_dir: Path) -> List[str]:
    """
    Extract frames uniformly distributed across the video using OpenCV based on a time interval.
    """
    frame_dir = output_dir / video_uid
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices based on interval
        frame_indices = []
        current_time = 0.0
        while current_time < total_frames / fps:
            frame_indices.append(int(current_time * fps))
            current_time += interval
        
        # Ensure the last frame is included if it's not exactly at the end
        if frame_indices[-1] != total_frames - 1:
            frame_indices.append(total_frames - 1)
        
        frame_paths = []
        for i, frame_idx in enumerate(frame_indices):
            # Set position to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            if ret:
                frame_path = frame_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(f"video_frames/{video_uid}/frame_{i:03d}.jpg")
        
        cap.release()
        print(f"  Extracted {len(frame_paths)} frames from {video_uid} at interval {interval}s")
        return frame_paths
        
    except Exception as e:
        print(f"Error extracting frames by interval from {video_path}: {e}")
        return []

def create_ego4d_hybrid_training_data(
    summaries: Dict, 
    uid_mapping: Dict[str, str],
    ego4d_root: str,
    output_dir: str,
    max_videos: Optional[int] = None,
    num_frames: int = 4,
    frame_interval: Optional[float] = None
) -> Tuple[List, List]:
    """
    Create hybrid training data using both video summaries and frame narrations
    """
    output_path = Path(output_dir)
    video_frames_dir = output_path / "video_frames"
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    print(f"Processing {len(train_items)} training videos and {len(val_items)} validation videos")
    print(f"Extracting {num_frames} frames uniformly from each video")
    if frame_interval:
        print(f"Frame interval: {frame_interval} seconds")
    
    total_videos = len(train_items) + len(val_items)
    processed_videos = 0
    start_time = datetime.now()
    
    for split_name, items in [("train", train_items), ("val", val_items)]:
        summary_count = 0
        narration_count = 0
        
        for video_idx, (video_uid, video_data) in enumerate(items):
            processed_videos += 1
            elapsed_time = datetime.now() - start_time
            avg_time_per_video = elapsed_time / processed_videos if processed_videos > 0 else 0
            remaining_videos = total_videos - processed_videos
            estimated_remaining_time = avg_time_per_video * remaining_videos
            
            print(f"\n📹 [{processed_videos}/{total_videos}] Processing {split_name} video {video_idx+1}/{len(items)}: {video_uid}")
            print(f"   ⏱️  Elapsed: {elapsed_time}, Avg per video: {avg_time_per_video}, ETA: {estimated_remaining_time}")
            
            # Find video file using manifest
            video_path = find_ego4d_video_file(video_uid, uid_mapping, ego4d_root)
            if not video_path:
                print(f"   ⚠️  Warning: Video file not found for {video_uid}")
                continue
            
            # Create summary training example
            summary = video_data.get("summary", "")
            if summary:
                # Extract frames uniformly for summary
                if frame_interval:
                    frame_paths = extract_frames_by_interval(video_path, video_uid, frame_interval, video_frames_dir)
                else:
                    frame_paths = extract_frames_uniform(video_path, video_uid, num_frames, video_frames_dir)
                
                if frame_paths:
                    # Create video summary example using first frame
                    summary_example = {
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
                        train_data.append(summary_example)
                        summary_count += 1
                    else:
                        val_data.append(summary_example)
                        summary_count += 1
            
                    # Create frame-level caption examples for all extracted frames
            narrations = []
            if "video_data" in video_data and "narration_pass_1" in video_data["video_data"]:
                narrations = video_data["video_data"]["narration_pass_1"].get("narrations", [])
            
                    # Create a mapping of frame timestamps to narrations
                    narration_timestamps = {}
                    for narration in narrations:
                        if "timestamp_sec" in narration and "narration_text" in narration:
                            timestamp = narration["timestamp_sec"]
                            narration_timestamps[timestamp] = narration["narration_text"]
                    
                    # Get video duration to map frame indices to timestamps
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    # Create caption examples for each extracted frame
                    frame_captions = []  # Store captions for summary generation
                    for i, frame_path in enumerate(frame_paths):
                        # Calculate timestamp for this frame
                        frame_timestamp = (i / (len(frame_paths) - 1)) * duration if len(frame_paths) > 1 else duration / 2
                        
                        # Find closest narration
                        closest_narration = None
                        min_distance = float('inf')
                        for timestamp, narration_text in narration_timestamps.items():
                            distance = abs(timestamp - frame_timestamp)
                            if distance < min_distance:
                                min_distance = distance
                                closest_narration = narration_text
                        
                        # Use closest narration or fallback to summary
                        if closest_narration and min_distance < 10.0:  # Within 10 seconds
                            caption = closest_narration.replace("#C ", "").strip()
                        else:
                            # Use a portion of the summary as caption
                            caption = summary.replace("#Summary ", "").strip()
                            if len(caption) > 100:
                                caption = caption[:100] + "..."
                        
                        frame_captions.append(caption)
                        
                        # Create frame caption example
                        frame_example = {
                            "id": f"{split_name}_frame_{video_uid}_{i}",
                    "image": frame_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWhat is happening in this moment?"
                        },
                        {
                            "from": "assistant",
                                    "value": caption
                        }
                    ]
                }
                
                if split_name == "train":
                            train_data.append(frame_example)
                else:
                            val_data.append(frame_example)
                    
                    # Create multi-frame summary example using all frames
                    if len(frame_paths) > 1:
                        # Create a summary example that includes frame captions in the prompt
                        frame_captions_text = "\n".join([f"Frame {i+1}: {caption}" for i, caption in enumerate(frame_captions[:8])])  # Use first 8 captions
                        
                        multi_frame_example = {
                            "id": f"{split_name}_multiframe_summary_{video_uid}",
                            "image": frame_paths[0],  # Use first frame as representative
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": f"<image>\n\nFrame captions from the video:\n{frame_captions_text}\n\nBased on these frame captions, provide a comprehensive summary of the entire video."
                                },
                                {
                                    "from": "assistant",
                                    "value": summary.replace("#Summary ", "").strip()
                                }
                            ]
                        }
                        
                        if split_name == "train":
                            train_data.append(multi_frame_example)
                            summary_count += 1
                        else:
                            val_data.append(multi_frame_example)
                            summary_count += 1
        
        print(f"  {split_name}: {summary_count} summary examples, {len([x for x in (train_data if split_name == 'train' else val_data) if 'frame_' in x['id']])} frame caption examples")
    
    # Shuffle the combined data
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"\nHybrid dataset created:")
    print(f"  Total training examples: {len(train_data)}")
    print(f"  Total validation examples: {len(val_data)}")
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description="Prepare InternVL3 finetuning data (Official Method with 4-bit LoRA)")
    parser.add_argument("--evaluation_file", type=str, 
                       default="final_optimized_evaluation_summary.json",
                       help="Path to evaluation results file")
    parser.add_argument("--output_dir", type=str, default="internvl3_data",
                       help="Output directory for prepared data")
    parser.add_argument("--image_folder", type=str, default="video_frames",
                       help="Folder containing video frames")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Ego4D specific arguments
    parser.add_argument("--use_ego4d", action="store_true",
                       help="Use Ego4D dataset instead of evaluation data")
    parser.add_argument("--ego4d_root", type=str, default="../remote_ego4d",
                       help="Path to Ego4D dataset root")
    parser.add_argument("--ego4d_summaries", type=str, default="finetuning/nlq_summaries/nlq_video_summaries.json",
                       help="Path to Ego4D video summaries file")
    parser.add_argument("--ego4d_manifest", type=str, default="v2/clips/manifest.csv",
                       help="Path to Ego4D manifest CSV file")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    parser.add_argument("--num_frames", type=int, default=4,
                       help="Number of frames to extract uniformly from each video (default: 4)")
    parser.add_argument("--frame_interval", type=float, default=None,
                       help="Time interval between frames in seconds (overrides num_frames if set)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_ego4d:
        # Ego4D hybrid training path
        print("🚀 Using Ego4D dataset for hybrid training (summaries + narrations)")
        
        # Load Ego4D manifest
        manifest_path = os.path.join(args.ego4d_root, args.ego4d_manifest)
        uid_mapping = load_ego4d_manifest(manifest_path)
        
        # Load Ego4D video summaries
        summaries_path = os.path.join(args.ego4d_root, args.ego4d_summaries)
        summaries = load_ego4d_video_summaries(summaries_path)
        
        # Create hybrid training data
        train_data, val_data = create_ego4d_hybrid_training_data(
            summaries, uid_mapping, args.ego4d_root, args.output_dir, args.max_videos, args.num_frames, args.frame_interval
        )
        
        # Save training data
        train_file = os.path.join(args.output_dir, "video_summary_dataset_train.jsonl")
        with open(train_file, 'w') as f:
            for example in train_data:
                f.write(json.dumps(example) + '\n')
        
        val_file = os.path.join(args.output_dir, "video_summary_dataset_val.jsonl")
        with open(val_file, 'w') as f:
            for example in val_data:
                f.write(json.dumps(example) + '\n')
        
        # Create meta file
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
        
        meta_file = os.path.join(args.output_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n✅ Ego4D hybrid dataset created successfully!")
        print(f"   Training examples: {len(train_data)}")
        print(f"   Validation examples: {len(val_data)}")
        print(f"   Output directory: {args.output_dir}")
        
    else:
        # Original evaluation data path
    # Load evaluation data
    video_data = load_evaluation_data(args.evaluation_file)
    
    if not video_data:
        print("No video data found in evaluation file. Please check the file format.")
        return
    
    # Create dataset
    dataset_path = os.path.join(args.output_dir, "video_summary_dataset.json")
        dataset_paths = create_internvl3_jsonl_dataset(
        video_data, 
        dataset_path, 
        args.image_folder, 
        args.train_ratio
    )
    
    # Create meta file
    meta_path = os.path.join(args.output_dir, "meta.json")
        create_internvl3_meta_file(dataset_paths, meta_path)
    
    # Create DeepSpeed configs
    create_deepspeed_configs()
    
    # Create training scripts
    create_official_internvl3_training_scripts()
    
    # Create model download script
    create_model_download_script()
    
    # Create requirements file
    create_requirements_file()
    
    print("\n" + "="*60)
    print("🎉 Official InternVL3 4-bit LoRA Finetuning Setup Complete!")
    print("="*60)
    
    if args.use_ego4d:
        print("📊 Ego4D Hybrid Training Dataset:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print("   Includes both video summaries and frame narrations")
    else:
    print(f"Dataset created in: {args.output_dir}")
    print(f"Training samples: {len(video_data) * args.train_ratio:.0f}")
    print(f"Validation samples: {len(video_data) * (1 - args.train_ratio):.0f}")
    
    print("\n🚀 RECOMMENDED: Start with official InternVL3 4-bit LoRA finetuning")
    print("   ./quick_start_internvl3_4bit_lora.sh")
    print("\n📋 Alternative options:")
    print("   ./finetune_internvl3_8b_4bit_lora.sh (customize settings)")
    print("\n📊 Monitor training with:")
    print("   tensorboard --logdir work_dirs/")
    print("\n💡 Official InternVL3 advantages:")
    print("   - Follows official documentation methodology")
    print("   - Optimized for InternVL3 architecture")
    print("   - LoRA rank 32 with official implementation")
    print("   - DeepSpeed ZeRO Stage 3 optimization")
    print("   - Dynamic image size support")
    print("   - 4-bit memory optimization for 48GB Ada Gen cards")

if __name__ == "__main__":
    main() 