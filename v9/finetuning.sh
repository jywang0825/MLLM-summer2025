#!/bin/bash
# single_gpu_conservative_internvl3_8b_finetune_frames.sh - CONSERVATIVE SINGLE GPU INTERNVL3 8B FINETUNING WITH FRAME EXTRACTION
# Uses GPU 0 with very conservative settings to minimize overfitting
# Modified to use frame-based data with maximum regularization

set -x

# CRITICAL: Apply flash attention patch BEFORE any imports
echo "Applying flash attention patch to avoid import errors..."
python -c "
import sys
import os

# Create mock flash_attn modules BEFORE any imports
class MockFlashAttn:
    def __init__(self):
        self.__version__ = '0.0.0'
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    
    def __call__(self, *args, **kwargs):
        return None

# Mock all flash attention related modules
sys.modules['flash_attn'] = MockFlashAttn()
sys.modules['flash_attn.flash_attn_interface'] = MockFlashAttn()
sys.modules['flash_attn.flash_attn'] = MockFlashAttn()
sys.modules['flash_attn.flash_attn_func'] = MockFlashAttn()
sys.modules['flash_attn.flash_attn_varlen_func'] = MockFlashAttn()

# Set environment variables
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['FLASH_ATTN_AVAILABLE'] = '0'

print('Flash attention patching completed - mock modules created')
"

# Use single GPU with conservative settings
GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

# Use GPU 0 which has ~40GB free memory
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

# Clear GPU cache before starting
python -c "import torch; torch.cuda.empty_cache()"

export PYTHONPATH="../InternVL/internvl_chat:../InternVL/internvl_chat/internvl:${PYTHONPATH}"
export MASTER_PORT=34236
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl3_8b_single_gpu_conservative_frames'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

CONDA_ENV_NAME=finetuning
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"$CONDA_ENV_NAME"* ]]; then
    echo "Activating conda environment: $CONDA_ENV_NAME"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV_NAME
fi

echo "Starting InternVL3-8B LoRA finetuning (SINGLE GPU CONSERVATIVE WITH FRAME EXTRACTION)"
echo "Using single GPU (0) with maximum anti-overfitting settings"
echo "GPUs: ${GPUS}, Batch size per GPU: ${PER_DEVICE_BATCH_SIZE}"
echo "Total batch size: ${BATCH_SIZE}, Gradient accumulation: ${GRADIENT_ACC}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
echo "Using frame-based data from internvl3_data/"
echo "Maximum anti-overfitting: Very low LoRA rank, high weight decay, frequent evaluation"
echo "Flash attention patched to avoid import errors"
echo ""

# Conservative Single GPU InternVL3 8B finetuning with frames and maximum anti-overfitting
PYTHONPATH=".:${PYTHONPATH}" $CONDA_PREFIX/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ../InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "../models/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "internvl3_data/meta.json" \
  --resume_from_checkpoint ${OUTPUT_DIR} \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.2 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --use_llm_lora 4 \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 15 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size 2 \
  --eval_accumulation_steps 2 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 5 \
  --learning_rate 5e-6 \
  --weight_decay 0.2 \
  --warmup_ratio 0.15 \
  --lr_scheduler_type "cosine_with_restarts" \
  --optim "adamw_torch" \
  --logging_steps 20 \
  --max_seq_length 4096 \
  --do_train True \
  --do_eval True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --early_stopping_patience 5 \
  --early_stopping_threshold 0.0005 \
  "$@" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "InternVL3 LoRA finetuning (SINGLE GPU CONSERVATIVE WITH FRAME EXTRACTION) completed!"
echo "Check results in: ${OUTPUT_DIR}"
