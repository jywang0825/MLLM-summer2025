#!/bin/bash

# Install LLaVA-NeXT dependencies for video captioning
echo "Installing LLaVA-NeXT dependencies..."

# Install LLaVA-NeXT
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# Install additional dependencies for video processing
pip install decord
pip install av
pip install transformers>=4.34.0
pip install accelerate>=0.20.3
pip install sentencepiece

# Install other required packages
pip install torch torchvision torchaudio
pip install pillow
pip install tqdm
pip install pandas

echo "LLaVA-NeXT installation completed!"
echo "You can now run the video captioning script with:"
echo "python3 ego4d_aks_caption_uniform_llava_video_7b_qwen2.py" 