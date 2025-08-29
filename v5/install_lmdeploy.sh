#!/bin/bash

# Install LMDeploy for optimized LLaVA inference
echo "Installing LMDeploy..."

# Install LMDeploy
pip install lmdeploy

# Install additional dependencies for video processing
pip install av
pip install decord

# Install LLaVA dependencies
pip install llava-torch
pip install transformers>=4.34.0
pip install accelerate>=0.20.3
pip install sentencepiece

echo "LMDeploy installation completed!"
echo "You can now run the optimized script with:"
echo "python3 ego4d_aks_caption_uniform_llava_video_7b_qwen2.py" 