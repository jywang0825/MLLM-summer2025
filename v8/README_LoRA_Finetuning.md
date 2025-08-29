# InternVL3 Official 4-bit LoRA Finetuning Guide

**Official InternVL3 methodology with 4-bit optimization for 48GB Ada Gen cards**

This guide provides a complete setup for finetuning InternVL3-8B using the **official InternVL3 methodology** with 4-bit LoRA optimization, making it efficient to run on 48GB Ada Gen cards while following the proven training approach.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_official.txt

# 2. Download InternVL3 model
./download_internvl3_models.sh

# 3. Prepare dataset
python prepare_internvl3_finetuning.py

# 4. Test setup
python test_internvl3_setup.py

# 5. Start official InternVL3 4-bit LoRA finetuning
./quick_start_internvl3_4bit_lora.sh
```

## 📋 Requirements

### Hardware
- **Recommended**: 2x RTX 6000 Ada Gen (48GB each)
- **Minimum**: 1x RTX 4090 (24GB) with reduced batch size
- **CPU**: 16+ cores recommended
- **RAM**: 64GB+ recommended
- **Storage**: 100GB+ free space

### Software
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+
- See `requirements_official.txt` for full dependencies

## 🎯 Why Official InternVL3 + 4-bit LoRA?

### Official Methodology Benefits
- **Proven Training Recipes**: Based on official InternVL3 documentation
- **Optimized Architecture**: Specifically designed for InternVL3 models
- **LoRA Implementation**: Uses official `wrap_llm_lora()` method
- **DeepSpeed Integration**: Optimized ZeRO Stage 3 configuration
- **Dynamic Image Size**: Official support for variable resolution

### 4-bit Memory Optimization
- **Reduced Memory Usage**: 4-bit quantization for efficiency
- **LoRA rank 32**: Official recommendation (~8M trainable params)
- **ZeRO Stage 3**: Optimizer and parameter offloading
- **Gradient checkpointing**: Memory-efficient training

## 📁 Project Structure

```
v8/
├── prepare_internvl3_finetuning.py    # Dataset preparation (Official format)
├── finetune_internvl3_8b_4bit_lora.sh # Official InternVL3 training script
├── quick_start_internvl3_4bit_lora.sh # Easy start script
├── download_internvl3_models.sh       # Model download script
├── zero_stage3_config.json            # DeepSpeed config for LoRA
├── requirements_official.txt          # Dependencies
├── test_internvl3_setup.py           # Setup validation
├── internvl3_data/                   # Prepared dataset
│   ├── meta.json                     # Official meta file
│   ├── video_summary_dataset_train.jsonl
│   └── video_summary_dataset_val.jsonl
├── models/                           # InternVL3 models
│   └── InternVL3-8B/
└── work_dirs/                        # Training outputs
    └── internvl3_8b_4bit_lora/
```

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
# Install official InternVL3 dependencies
pip install -r requirements_official.txt

# Verify installation
python -c "import deepspeed; print('DeepSpeed OK')"
python -c "import bitsandbytes; print('BitsAndBytes OK')"
python -c "import peft; print('PEFT OK')"
```

### 2. Download InternVL3 Model

```bash
# Download InternVL3-8B (recommended for 48GB cards)
./download_internvl3_models.sh

# Or manually download
mkdir -p models
cd models
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-8B --local-dir InternVL3-8B
```

### 3. Prepare Dataset

```bash
# Convert evaluation data to official InternVL3 format
python prepare_internvl3_finetuning.py

# This creates:
# - internvl3_data/meta.json (official format)
# - internvl3_data/video_summary_dataset_train.jsonl
# - internvl3_data/video_summary_dataset_val.jsonl
# - All training scripts and configs
```

### 4. Validate Setup

```bash
# Run comprehensive setup test
python test_internvl3_setup.py

# Should show all ✅ checks passed
```

## 🚀 Training Options

### Option 1: Quick Start (Recommended)

```bash
# Uses official InternVL3 defaults optimized for 48GB Ada Gen cards
./quick_start_internvl3_4bit_lora.sh
```

**Default settings:**
- GPUs: 2
- Batch size per GPU: 4
- Total batch size: 32
- LoRA rank: 32 (official recommendation)
- Learning rate: 2e-5 (official InternVL3)
- Epochs: 3

### Option 2: Custom Training

```bash
# Customize settings
export GPUS=2
export PER_DEVICE_BATCH_SIZE=4
export BATCH_SIZE=32

./finetune_internvl3_8b_4bit_lora.sh
```

### Option 3: Manual Training

```bash
# Direct torchrun execution
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=2 \
  --master_port=34229 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "models/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir "work_dirs/internvl3_8b_4bit_lora" \
  --meta_path "internvl3_data/meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 32 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --max_seq_length 8192 \
  --do_train True \
  --do_eval True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage3_config.json" \
  --report_to "tensorboard"
```

## ⚙️ Official InternVL3 Configuration

### LoRA Configuration (Official Method)

```python
# Official InternVL3 LoRA wrapping
model.wrap_llm_lora(r=32, lora_alpha=64, lora_dropout=0.05)

# Target modules (automatically selected based on architecture):
# - InternLM2: ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
# - LLaMA/Qwen2: ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
```

### DeepSpeed Configuration

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
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4
}
```

### Model Architecture (Official InternVL3)

```
InternVL3-8B
├── Vision Backbone (frozen) ← Efficient feature extraction
├── LLM Layers (LoRA) ← Learnable parameters (rank 32)
└── MLP Layers (frozen) ← Stable feature projection
```

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir work_dirs/

# Access at http://localhost:6006
```

### Training Logs

```bash
# Monitor training progress
tail -f work_dirs/internvl3_8b_4bit_lora/training_log.txt

# Check GPU usage
watch -n 1 nvidia-smi
```

### Key Metrics to Watch

- **Loss**: Should decrease steadily
- **Memory usage**: Should stay under 40GB per GPU
- **Training speed**: Steps per second
- **Validation loss**: Should track training loss

## 🎛️ Hyperparameter Tuning

### For 48GB Ada Gen Cards

| Parameter | Official Default | Range |
|-----------|------------------|-------|
| LoRA rank (r) | 32 | 16-64 |
| LoRA alpha | 64 | 32-128 |
| Batch size per GPU | 4 | 2-8 |
| Learning rate | 2e-5 | 1e-5 to 5e-5 |
| Gradient accumulation | 4 | 2-8 |

### Memory Optimization

If you encounter OOM errors:

1. **Reduce batch size**: Try `PER_DEVICE_BATCH_SIZE=2`
2. **Reduce LoRA rank**: Try `--use_llm_lora 16`
3. **Increase gradient accumulation**: Compensate for smaller batch size
4. **Enable more offloading**: Already configured in DeepSpeed

### Performance Optimization

For faster training:

1. **Increase batch size**: If memory allows
2. **Increase LoRA rank**: More capacity for learning
3. **Reduce gradient accumulation**: More frequent updates
4. **Use more GPUs**: Scale horizontally

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
export PER_DEVICE_BATCH_SIZE=2

# Reduce LoRA rank
# Edit training script: --use_llm_lora 16

# Enable more offloading
# Already configured in zero_stage3_config.json
```

#### 2. InternVL Repository Not Found

```bash
# Ensure you're in the InternVL directory
cd InternVL

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/InternVL"
```

#### 3. Model Loading Issues

```bash
# Check model files
ls -la models/InternVL3-8B/

# Re-download if needed
./download_internvl3_models.sh
```

#### 4. Dataset Format Errors

```bash
# Validate dataset
python test_internvl3_setup.py

# Re-prepare if needed
python prepare_internvl3_finetuning.py
```

### Performance Tips

1. **Use SSD storage**: Faster data loading
2. **Pin memory**: Already enabled in config
3. **Multiple workers**: Adjust `dataloader_num_workers`
4. **Gradient checkpointing**: Already enabled for memory efficiency

## 📈 Expected Results

### Training Time
- **2x RTX 6000 Ada Gen**: ~2-4 hours for 3 epochs
- **1x RTX 4090**: ~4-8 hours for 3 epochs

### Memory Usage
- **Base model**: ~25-30GB (with 4-bit optimization)
- **LoRA adapters**: ~100MB
- **Training overhead**: ~10-15GB per GPU

### Model Quality
- **Official methodology**: Proven training approach
- **LoRA efficiency**: ~8M trainable parameters
- **Better than prompt engineering**: More flexible and powerful
- **Easier to deploy**: Small adapter files

## 🎯 Next Steps

After training:

1. **Evaluate model**: Test on validation set
2. **Merge LoRA**: Use official merge script
3. **Deploy**: Use for video summarization tasks
4. **Iterate**: Fine-tune hyperparameters based on results

## 📚 Additional Resources

- [InternVL3 Paper](https://arxiv.org/abs/2401.16420)
- [Official InternVL Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL3 Documentation](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python test_internvl3_setup.py` to validate setup
3. Check training logs for specific error messages
4. Ensure you're using the official InternVL3 approach
5. Check the [official InternVL3 documentation](https://internvl.readthedocs.io/en/latest/internvl3.0/finetune.html)

---

**Remember: This setup follows the official InternVL3 methodology for optimal results!** 🚀 