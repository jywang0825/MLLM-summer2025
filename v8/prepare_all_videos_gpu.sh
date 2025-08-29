#!/bin/bash
# prepare_all_videos_gpu.sh - Prepare InternVL3 data from ALL videos with GPU acceleration

set -x

# Configuration
NUM_FRAMES=${NUM_FRAMES:-8}  # Number of frames to extract uniformly from each video
FRAME_INTERVAL=${FRAME_INTERVAL:-}  # Time interval between frames (optional)

# Force GPU-only execution
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1

echo "🚀 Preparing InternVL3 finetuning data from ALL videos with GPU acceleration"
echo "Number of frames per video: ${NUM_FRAMES}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES}"
if [ ! -z "$FRAME_INTERVAL" ]; then
    echo "Frame interval: ${FRAME_INTERVAL} seconds"
fi

# Activate conda environment
CONDA_ENV_NAME=finetuning
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"$CONDA_ENV_NAME"* ]]; then
    echo "Activating conda environment: $CONDA_ENV_NAME"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV_NAME
fi

# Verify GPU setup
echo "🔍 Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
"

# Create log file with timestamp
LOG_FILE="internvl3_data_preparation_$(date +%Y%m%d_%H%M%S).log"
echo "📋 Log file: $LOG_FILE"

# Prepare data from ALL videos with GPU acceleration
echo "📊 Starting data preparation from ALL videos..."
echo "🔄 This will run in the background. Monitor progress with:"
echo "   tail -f $LOG_FILE"
echo "   or check GPU usage with: nvidia-smi"

if [ ! -z "$FRAME_INTERVAL" ]; then
    # Use frame interval
    nohup python3 prepare_internvl3_finetuning.py \
        --use_ego4d \
        --ego4d_root ../remote_ego4d \
        --frame_interval ${FRAME_INTERVAL} \
        --output_dir internvl3_data \
        > $LOG_FILE 2>&1 &
else
    # Use number of frames (process ALL videos)
    nohup python3 prepare_internvl3_finetuning.py \
        --use_ego4d \
        --ego4d_root ../remote_ego4d \
        --num_frames ${NUM_FRAMES} \
        --output_dir internvl3_data \
        > $LOG_FILE 2>&1 &
fi

# Get the background process ID
PID=$!
echo "✅ Data preparation started in background (PID: $PID)"
echo "📊 Monitor progress with:"
echo "   tail -f $LOG_FILE"
echo "   ps aux | grep $PID"
echo "   nvidia-smi -l 1"
echo ""
echo "🛑 To stop the process: kill $PID"
echo "📋 Log file: $LOG_FILE" 