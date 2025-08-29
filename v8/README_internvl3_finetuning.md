# InternVL3 Supervised Finetuning Setup

## 🎯 Overview

This repository contains a complete setup for supervised finetuning of InternVL3 models on video summarization tasks. The setup includes data preparation, frame extraction, training scripts, and evaluation tools.

## 📁 Files Created

### Core Scripts
- **`prepare_internvl3_finetuning.py`** - Converts evaluation data to InternVL3 training format
- **`extract_video_frames.py`** - Extracts frames from Ego4D videos for training
- **`test_internvl3_setup.py`** - Tests the complete setup before training

### Training Scripts
- **`finetune_internvl3_8b_basic.sh`** - Full finetuning script (requires more GPU memory)
- **`finetune_internvl3_8b_lora.sh`** - LoRA finetuning script (memory efficient)

### Configuration Files
- **`zero_stage1_config.json`** - DeepSpeed config for full finetuning
- **`zero_stage3_config.json`** - DeepSpeed config for LoRA finetuning

### Documentation
- **`internvl3_finetuning_guide.md`** - Comprehensive finetuning guide
- **`supervised_finetuning_guide.md`** - Step-by-step supervised finetuning workflow

## 🚀 Quick Start

### 1. Prepare Your Data

```bash
# Convert evaluation data to training format
python prepare_internvl3_finetuning.py --evaluation_file final_optimized_evaluation_summary.json

# Extract video frames from Ego4D dataset
python extract_video_frames.py --evaluation_file final_optimized_evaluation_summary.json

# Test the complete setup
python test_internvl3_setup.py
```

### 2. Start Training

```bash
# For LoRA finetuning (recommended for most users)
./finetune_internvl3_8b_lora.sh

# For full finetuning (requires more GPU memory)
./finetune_internvl3_8b_basic.sh
```

### 3. Monitor Training

```bash
# Monitor with tensorboard
tensorboard --logdir work_dirs/

# Monitor GPU usage
nvidia-smi -l 1

# Check training logs
tail -f work_dirs/internvl3_8b_finetune_lora/training_log.txt
```

## 📊 Hardware Requirements

| Model | GPU Memory | GPUs | Training Time |
|-------|------------|------|---------------|
| InternVL3-8B | 24GB+ | 2-4 | 2-4 hours |
| InternVL3-8B (LoRA) | 16GB+ | 1-2 | 1-2 hours |

## 🔧 Configuration

### Training Parameters

**LoRA Finetuning (Recommended):**
- Learning rate: 2e-5
- Batch size: 64 (4 per GPU × 4 GPUs × 4 accumulation)
- Epochs: 1-2
- LoRA rank: 16
- Sequence length: 8192

**Full Finetuning:**
- Learning rate: 2e-5
- Batch size: 128 (4 per GPU × 4 GPUs × 8 accumulation)
- Epochs: 1-3
- Sequence length: 16384

### Data Format

The training data is converted to InternVL3 conversation format:

```json
[
  {
    "id": "train_video_001",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease provide a concise summary of this video."
      },
      {
        "from": "assistant",
        "value": "A person is cooking in the kitchen, preparing a meal."
      }
    ],
    "image": "video_frames/video_001/frame_000.jpg"
  }
]
```

## 📈 Evaluation

### Training Metrics
Monitor these during training:
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should follow training loss
- **Learning Rate**: Should follow cosine schedule
- **GPU Memory**: Should be stable

### Evaluation Metrics
Use your existing evaluation pipeline:

```python
# Load your evaluation script
from evaluate_with_clair_simple import evaluate_model

# Evaluate finetuned model
results = evaluate_model(
    model_path="work_dirs/internvl3_8b_finetune_lora/checkpoint-200",
    test_data="test_summaries_minimal.json"
)
```

## 🛠️ Troubleshooting

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
   python test_internvl3_setup.py
   ```

3. **Frame Extraction Issues**
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
tail -f work_dirs/internvl3_8b_finetune_lora/training_log.txt

# Check dataset format
python -c "
import json
data = json.load(open('internvl3_data/video_summary_dataset_train.json'))
print(f'Dataset size: {len(data)}')
print(f'Sample conversation: {data[0]}')
"
```

## 📚 Detailed Guides

- **[InternVL3 Finetuning Guide](internvl3_finetuning_guide.md)** - Comprehensive guide covering all aspects of finetuning
- **[Supervised Finetuning Guide](supervised_finetuning_guide.md)** - Step-by-step workflow for supervised finetuning

## 🎯 Best Practices

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

## 🔄 Workflow Summary

1. **Data Preparation** → `prepare_internvl3_finetuning.py`
2. **Frame Extraction** → `extract_video_frames.py`
3. **Setup Testing** → `test_internvl3_setup.py`
4. **Training** → `finetune_internvl3_8b_lora.sh`
5. **Monitoring** → Tensorboard + training logs
6. **Evaluation** → Your existing evaluation pipeline

## 📖 Resources

- [InternVL3 Paper](https://huggingface.co/papers/2504.10479)
- [InternVL GitHub](https://github.com/OpenGVLab/InternVL)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 🎉 Next Steps

1. **Start with LoRA finetuning** to test your setup
2. **Experiment with hyperparameters** to find optimal settings
3. **Use full finetuning** for best performance
4. **Evaluate on your test set** and compare with baselines
5. **Iterate and improve** based on results

---

**Happy Finetuning! 🚀**

This setup should help you successfully finetune InternVL3 on your video summarization task. Start with the quick start section and gradually explore the advanced techniques as needed. 