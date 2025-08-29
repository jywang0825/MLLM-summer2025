# InternVL3 Finetuning Guide

## Overview

InternVL3 is an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance. This guide will help you finetune InternVL3 models for your specific use cases.

## Model Architecture

InternVL3 follows the "ViT-MLP-LLM" paradigm:
- **Vision Part**: InternViT-300M-448px-V2_5 (for 1B-14B models) or InternViT-6B-448px-V2_5 (for 38B-78B models)
- **Language Part**: Qwen2.5 series or InternLM3
- **MLP Projector**: Randomly initialized MLP projector connecting vision and language components

Key features:
- Variable Visual Position Encoding (V2PE) for better long context understanding
- Dynamic resolution strategy (448×448 pixel tiles)
- Multi-image and video data support
- Pixel unshuffle operation reducing visual tokens to 1/4

## Available Models

| Model Name | Vision Part | Language Part | HF Link |
|------------|-------------|---------------|---------|
| InternVL3-1B | InternViT-300M-448px-V2_5 | Qwen2.5-0.5B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-1B) |
| InternVL3-2B | InternViT-300M-448px-V2_5 | Qwen2.5-1.5B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-2B) |
| **InternVL3-8B** | InternViT-300M-448px-V2_5 | Qwen2.5-7B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-8B) |
| InternVL3-9B | InternViT-300M-448px-V2_5 | internlm3-8b-instruct | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-9B) |
| InternVL3-14B | InternViT-300M-448px-V2_5 | Qwen2.5-14B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-14B) |
| InternVL3-38B | InternViT-6B-448px-V2_5 | Qwen2.5-32B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-38B) |
| InternVL3-78B | InternViT-6B-448px-V2_5 | Qwen2.5-72B | [🤗 link](https://huggingface.co/OpenGVLab/InternVL3-78B) |

## Prerequisites

### Hardware Requirements

| Model Size | Minimum GPU Memory | Recommended GPU Memory | GPUs |
|------------|-------------------|----------------------|------|
| 1B-2B | 8GB | 16GB | 1-2 |
| 8B-9B | 16GB | 24GB | 2-4 |
| 14B | 24GB | 32GB | 4-8 |
| 38B | 48GB | 64GB | 8-16 |
| 78B | 80GB | 120GB | 16+ |

### Software Requirements

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.37.2
pip install accelerate
pip install deepspeed
pip install datasets
pip install pillow
pip install numpy

# For distributed training
pip install torchvision
pip install tensorboard

# Optional: For better performance
pip install flash-attn
pip install xformers
```

## Setup

### 1. Clone the InternVL Repository

```bash
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
pip install -e .
```

### 2. Download the Model

You already have InternVL3-8B in your `models/InternVL3-8B/` directory. For other models:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3-8B

# Or using huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download(repo_id="OpenGVLab/InternVL3-8B", local_dir="./models/InternVL3-8B")
```

## Data Preparation

### 1. Data Format

InternVL3 supports multiple data formats. Here's the standard format for instruction tuning:

```json
[
  {
    "id": "unique_id",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease describe this image in detail."
      },
      {
        "from": "assistant", 
        "value": "This image shows a beautiful sunset over mountains..."
      }
    ],
    "image": "path/to/image.jpg"
  }
]
```

### 2. Create Your Dataset

For video summarization (your use case), create a dataset like this:

```python
import json
import os

def create_video_summary_dataset(video_data, output_path):
    """
    Create a dataset for video summarization finetuning
    
    Args:
        video_data: List of dicts with keys: video_id, frames, summary
        output_path: Path to save the dataset
    """
    dataset = []
    
    for item in video_data:
        # Create conversation format
        conversation = {
            "id": item["video_id"],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nPlease provide a concise summary of this video."
                },
                {
                    "from": "assistant",
                    "value": item["summary"]
                }
            ],
            "image": item["frames"][0]  # Use first frame as representative
        }
        dataset.append(conversation)
    
    # Save dataset
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return dataset

# Example usage
video_data = [
    {
        "video_id": "video_001",
        "frames": ["/path/to/frame1.jpg", "/path/to/frame2.jpg"],
        "summary": "A person is cooking in the kitchen, preparing a meal."
    }
]

create_video_summary_dataset(video_data, "video_summary_dataset.json")
```

### 3. Meta File

Create a meta file to specify your dataset:

```json
{
  "video_summary": {
    "train": "path/to/video_summary_dataset.json",
    "image_folder": "path/to/images"
  }
}
```

## Finetuning Scripts

### 1. Basic Finetuning Script

Create a finetuning script for InternVL3-8B:

```bash
#!/bin/bash
# finetune_internvl3_8b.sh

set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl3_8b_finetune'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# Finetuning configuration
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "models/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "path/to/your/meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
```

### 2. LoRA Finetuning (Memory Efficient)

For memory-efficient finetuning using LoRA:

```bash
#!/bin/bash
# finetune_internvl3_8b_lora.sh

set -x

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl3_8b_finetune_lora'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# LoRA finetuning configuration
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "models/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "path/to/your/meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
```

### 3. DeepSpeed Configuration

Create DeepSpeed configuration files:

**zero_stage1_config.json:**
```json
{
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4
}
```

**zero_stage3_config.json:**
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4
}
```

## Training Parameters

### Key Parameters Explained

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--learning_rate` | Learning rate for training | 1e-5 to 2e-5 |
| `--num_train_epochs` | Number of training epochs | 1-3 |
| `--per_device_train_batch_size` | Batch size per GPU | 1-4 (depends on GPU memory) |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 4-8 |
| `--max_seq_length` | Maximum sequence length | 8192-16384 |
| `--freeze_backbone` | Freeze vision backbone | True (for efficiency) |
| `--freeze_llm` | Freeze language model | False (for full finetuning) |
| `--use_llm_lora` | Use LoRA for LLM | 16 (for LoRA finetuning) |

### Memory Optimization Tips

1. **Use LoRA**: Reduces memory usage significantly
2. **Freeze backbone**: Freeze vision encoder to save memory
3. **Gradient checkpointing**: Enable with `--grad_checkpoint True`
4. **Mixed precision**: Use bf16 for faster training and less memory
5. **DeepSpeed ZeRO**: Use ZeRO-3 for large models

## Running the Training

### 1. Single Node Training

```bash
# Make script executable
chmod +x finetune_internvl3_8b.sh

# Run training
./finetune_internvl3_8b.sh
```

### 2. Multi-Node Training

```bash
# For SLURM systems
srun -p your_partition \
  --gres=gpu:8 \
  --nodes=2 \
  --ntasks=16 \
  --ntasks-per-node=8 \
  --cpus-per-task=10 \
  --kill-on-bad-exit=1 \
  --quotatype=reserved \
  python -u internvl/train/internvl_chat_finetune.py \
  [your training arguments]
```

### 3. Monitor Training

```bash
# Monitor with tensorboard
tensorboard --logdir work_dirs/internvl3_8b_finetune

# Monitor GPU usage
nvidia-smi -l 1

# Check training logs
tail -f work_dirs/internvl3_8b_finetune/training_log.txt
```

## Evaluation and Testing

### 1. Load and Test Your Finetuned Model

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load your finetuned model
model_path = "work_dirs/internvl3_8b_finetune/checkpoint-200"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Test with an image
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path, max_num=12):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Test inference
image_path = "path/to/test/image.jpg"
pixel_values = load_image(image_path).to(torch.bfloat16).cuda()

question = "<image>\nPlease describe this image in detail."
generation_config = dict(max_new_tokens=512, do_sample=False)

response, history = model.chat(
    tokenizer, 
    pixel_values, 
    question, 
    generation_config, 
    history=None, 
    return_history=True
)

print(f"User: {question}")
print(f"Assistant: {response}")
```

### 2. Evaluate on Your Test Set

```python
import json
from tqdm import tqdm

def evaluate_model(model, tokenizer, test_data):
    """Evaluate model on test dataset"""
    results = []
    
    for item in tqdm(test_data):
        # Load image
        pixel_values = load_image(item["image"]).to(torch.bfloat16).cuda()
        
        # Generate response
        question = "<image>\nPlease describe this image in detail."
        generation_config = dict(max_new_tokens=512, do_sample=False)
        
        response, _ = model.chat(
            tokenizer, 
            pixel_values, 
            question, 
            generation_config, 
            history=None, 
            return_history=False
        )
        
        results.append({
            "id": item["id"],
            "ground_truth": item["summary"],
            "prediction": response,
            "image": item["image"]
        })
    
    return results

# Load test data
with open("test_data.json", "r") as f:
    test_data = json.load(f)

# Evaluate
results = evaluate_model(model, tokenizer, test_data)

# Save results
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Best Practices

### 1. Data Quality
- Ensure high-quality, diverse training data
- Clean and validate your dataset
- Use appropriate data augmentation

### 2. Training Strategy
- Start with a small learning rate (1e-5 to 2e-5)
- Use warmup and cosine learning rate scheduling
- Monitor training loss and validation metrics
- Use early stopping to prevent overfitting

### 3. Memory Management
- Use LoRA for large models if memory is limited
- Enable gradient checkpointing
- Use mixed precision training (bf16)
- Optimize batch size and gradient accumulation

### 4. Model Selection
- Choose model size based on your task complexity and hardware
- Consider using smaller models for faster iteration
- Use larger models for better performance when resources allow

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Use LoRA finetuning
   - Enable gradient checkpointing
   - Use DeepSpeed ZeRO-3

2. **Training Loss Not Decreasing**
   - Check learning rate
   - Verify data format
   - Check model initialization
   - Monitor gradient norms

3. **Slow Training**
   - Use mixed precision (bf16)
   - Optimize data loading
   - Use flash attention if available
   - Increase batch size if memory allows

### Debugging Tips

```bash
# Check GPU memory usage
nvidia-smi

# Monitor training progress
tail -f work_dirs/internvl3_8b_finetune/training_log.txt

# Check model parameters
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('models/InternVL3-8B', trust_remote_code=True)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"
```

## Next Steps

1. **Start with a small dataset** to test your setup
2. **Experiment with different hyperparameters** to find optimal settings
3. **Use your existing video summarization data** from the evaluation results
4. **Consider using SWIFT or XTurner** for easier finetuning workflows
5. **Explore advanced techniques** like Mixed Preference Optimization (MPO)

## Resources

- [InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL Documentation](https://internvl.readthedocs.io/en/latest/)
- [SWIFT Finetuning Framework](https://github.com/modelscope/ms-swift)
- [XTurner Finetuning Framework](https://github.com/InternLM/xtuner)
- [InternVL3 Paper](https://huggingface.co/papers/2504.10479)

This guide should help you get started with finetuning InternVL3 for your video summarization task. Start with the basic finetuning script and adjust parameters based on your specific requirements and hardware constraints. 