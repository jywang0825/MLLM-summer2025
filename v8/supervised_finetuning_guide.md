# InternVL3 Supervised Finetuning Guide

## Overview

This guide provides a complete workflow for supervised finetuning of InternVL3 models on your video summarization task. We'll use your existing evaluation data to create a proper training dataset and finetune the model.

## Quick Start

### 1. Prepare Your Data

```bash
# Step 1: Prepare the finetuning dataset
python prepare_internvl3_finetuning.py --evaluation_file final_optimized_evaluation_summary.json

# Step 2: Extract video frames from Ego4D dataset
python extract_video_frames.py --evaluation_file final_optimized_evaluation_summary.json

# Step 3: Verify the setup
ls -la internvl3_data/
ls -la video_frames/
```

### 2. Start Training

```bash
# For full finetuning (requires more GPU memory)
./finetune_internvl3_8b_basic.sh

# For LoRA finetuning (memory efficient)
./finetune_internvl3_8b_lora.sh
```

### 3. Monitor Training

```bash
# Monitor with tensorboard
tensorboard --logdir work_dirs/

# Monitor GPU usage
nvidia-smi -l 1

# Check training logs
tail -f work_dirs/internvl3_8b_finetune_basic/training_log.txt
```

## Detailed Workflow

### Step 1: Data Preparation

The `prepare_internvl3_finetuning.py` script does the following:

1. **Loads your evaluation data** from `final_optimized_evaluation_summary.json`
2. **Converts to InternVL3 format** with conversations
3. **Splits into train/validation** sets (80/20 by default)
4. **Creates meta file** for training
5. **Generates DeepSpeed configs** and training scripts

```bash
python prepare_internvl3_finetuning.py \
    --evaluation_file final_optimized_evaluation_summary.json \
    --output_dir internvl3_data \
    --train_ratio 0.8 \
    --seed 42
```

### Step 2: Frame Extraction

The `extract_video_frames.py` script:

1. **Finds video files** in your mounted Ego4D dataset
2. **Extracts frames** at specified FPS (default: 1 fps)
3. **Updates dataset** with frame paths
4. **Creates frame index** for reference

```bash
python extract_video_frames.py \
    --evaluation_file final_optimized_evaluation_summary.json \
    --ego4d_root ~/remote_ego4d \
    --output_dir video_frames \
    --fps 1 \
    --max_frames 10
```

### Step 3: Training Configuration

#### Hardware Requirements

| Model | GPU Memory | GPUs | Training Time |
|-------|------------|------|---------------|
| InternVL3-8B | 24GB+ | 2-4 | 2-4 hours |
| InternVL3-8B (LoRA) | 16GB+ | 1-2 | 1-2 hours |

#### Training Parameters

**Full Finetuning:**
- Learning rate: 2e-5
- Batch size: 128 (4 per GPU × 4 GPUs × 8 accumulation)
- Epochs: 1-3
- Sequence length: 16384

**LoRA Finetuning:**
- Learning rate: 2e-5
- Batch size: 64 (4 per GPU × 4 GPUs × 4 accumulation)
- Epochs: 1-2
- LoRA rank: 16
- Sequence length: 8192

### Step 4: Training Execution

#### Basic Finetuning

```bash
# Adjust GPU settings based on your hardware
export GPUS=4
export PER_DEVICE_BATCH_SIZE=2

# Run training
./finetune_internvl3_8b_basic.sh
```

#### LoRA Finetuning (Recommended for most users)

```bash
# Adjust GPU settings
export GPUS=2
export PER_DEVICE_BATCH_SIZE=4

# Run LoRA training
./finetune_internvl3_8b_lora.sh
```

### Step 5: Evaluation and Testing

#### Load Your Finetuned Model

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load finetuned model
model_path = "work_dirs/internvl3_8b_finetune_basic/checkpoint-200"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
```

#### Test on New Videos

```python
from PIL import Image
import torchvision.transforms as transforms

def test_video_summary(model, tokenizer, video_frames):
    """Test model on video frames"""
    
    # Load and preprocess frames
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use first frame as representative
    frame = Image.open(video_frames[0]).convert('RGB')
    pixel_values = transform(frame).unsqueeze(0).to(torch.bfloat16).cuda()
    
    # Generate summary
    question = "<image>\nPlease provide a concise summary of this video."
    generation_config = dict(max_new_tokens=512, do_sample=False)
    
    response, _ = model.chat(
        tokenizer, 
        pixel_values, 
        question, 
        generation_config, 
        history=None, 
        return_history=False
    )
    
    return response

# Test on a video
video_frames = ["video_frames/video_001/frame_000.jpg"]
summary = test_video_summary(model, tokenizer, video_frames)
print(f"Generated summary: {summary}")
```

## Advanced Training Techniques

### 1. Mixed Precision Training

InternVL3 supports mixed precision training with bf16:

```bash
# Already enabled in scripts
--bf16 True
```

### 2. Gradient Checkpointing

Reduces memory usage by recomputing gradients:

```bash
# Already enabled in scripts
--grad_checkpoint True
```

### 3. Dynamic Resolution

InternVL3 supports dynamic resolution for better efficiency:

```bash
# Already enabled in scripts
--dynamic_image_size True
--use_thumbnail True
```

### 4. Data Packing

For efficient training with multiple images:

```bash
# Add to training script for packed training
--use_packed_ds True
--num_images_expected 40
--max_packed_tokens 8192
```

## Evaluation Metrics

### 1. Training Metrics

Monitor these during training:
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should follow training loss
- **Learning Rate**: Should follow cosine schedule
- **GPU Memory**: Should be stable

### 2. Evaluation Metrics

Use your existing evaluation pipeline:

```python
# Load your evaluation script
from evaluate_with_clair_simple import evaluate_model

# Evaluate finetuned model
results = evaluate_model(
    model_path="work_dirs/internvl3_8b_finetune_basic/checkpoint-200",
    test_data="test_summaries_minimal.json"
)

# Compare with baseline
print("Finetuned model results:", results)
```

### 3. BLEU Score Comparison

Compare your finetuned model with the baseline:

```python
import json

# Load baseline results
with open("final_optimized_evaluation_summary.json", "r") as f:
    baseline = json.load(f)

# Load finetuned results
with open("finetuned_evaluation_results.json", "r") as f:
    finetuned = json.load(f)

# Compare BLEU scores
print("Baseline BLEU-2:", baseline["results"]["v1_ego4d_aks_captions_optimized"]["metrics"]["bleu_2"])
print("Finetuned BLEU-2:", finetuned["metrics"]["bleu_2"])
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   export PER_DEVICE_BATCH_SIZE=1
   
   # Use LoRA instead
   ./finetune_internvl3_8b_lora.sh
   ```

2. **Training Loss Not Decreasing**
   ```bash
   # Check learning rate
   --learning_rate 1e-5  # Try lower learning rate
   
   # Check data format
   python -c "import json; data=json.load(open('internvl3_data/video_summary_dataset_train.json')); print(len(data))"
   ```

3. **Slow Training**
   ```bash
   # Use mixed precision
   --bf16 True
   
   # Reduce sequence length
   --max_seq_length 8192
   ```

4. **Frame Extraction Issues**
   ```bash
   # Check video file paths
   ls ~/remote_ego4d/videos/ | head -10
   
   # Check frame extraction
   ls video_frames/ | head -10
   ```

### Debugging Commands

```bash
# Check GPU memory
nvidia-smi

# Check training progress
tail -f work_dirs/internvl3_8b_finetune_basic/training_log.txt

# Check dataset format
python -c "
import json
data = json.load(open('internvl3_data/video_summary_dataset_train.json'))
print(f'Dataset size: {len(data)}')
print(f'Sample conversation: {data[0]}')
"

# Check model parameters
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('models/InternVL3-8B', trust_remote_code=True)
print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"
```

## Best Practices

### 1. Data Quality
- Ensure high-quality ground truth summaries
- Clean and validate your dataset
- Use consistent formatting

### 2. Training Strategy
- Start with LoRA finetuning for quick iteration
- Use full finetuning for best performance
- Monitor validation metrics
- Use early stopping to prevent overfitting

### 3. Memory Management
- Use LoRA for large models
- Enable gradient checkpointing
- Use mixed precision training
- Optimize batch size

### 4. Evaluation
- Evaluate on a held-out test set
- Compare with baseline models
- Monitor multiple metrics (BLEU, METEOR, ROUGE)
- Test on diverse video types

## Next Steps

1. **Start with LoRA finetuning** to test your setup
2. **Experiment with hyperparameters** to find optimal settings
3. **Use full finetuning** for best performance
4. **Evaluate on your test set** and compare with baselines
5. **Iterate and improve** based on results

## Resources

- [InternVL3 Paper](https://huggingface.co/papers/2504.10479)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

This supervised finetuning guide should help you successfully finetune InternVL3 on your video summarization task. Start with the quick start section and gradually explore the advanced techniques as needed. 