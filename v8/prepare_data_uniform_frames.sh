#!/bin/bash
# prepare_data_uniform_frames.sh - Prepare InternVL3 data with uniform frame extraction

set -x

# Configuration
NUM_FRAMES=${NUM_FRAMES:-4}  # Number of frames to extract uniformly from each video
MAX_VIDEOS=${MAX_VIDEOS:-100}  # Maximum number of videos to process
FRAME_INTERVAL=${FRAME_INTERVAL:-}  # Time interval between frames (optional)

echo "🚀 Preparing InternVL3 finetuning data with uniform frame extraction"
echo "Number of frames per video: ${NUM_FRAMES}"
echo "Maximum videos to process: ${MAX_VIDEOS}"
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

# Prepare data with uniform frame extraction
if [ ! -z "$FRAME_INTERVAL" ]; then
    # Use frame interval
    python3 prepare_internvl3_finetuning.py \
        --use_ego4d \
        --ego4d_root ../remote_ego4d \
        --max_videos ${MAX_VIDEOS} \
        --frame_interval ${FRAME_INTERVAL} \
        --output_dir internvl3_data
else
    # Use number of frames
    python3 prepare_internvl3_finetuning.py \
        --use_ego4d \
        --ego4d_root ../remote_ego4d \
        --max_videos ${MAX_VIDEOS} \
        --num_frames ${NUM_FRAMES} \
        --output_dir internvl3_data
fi

echo "✅ Data preparation completed!"
echo "📊 Check the generated files in internvl3_data/"
echo "🚀 Ready to start finetuning with: ./finetune_internvl3_8b_4bit_lora.sh" 