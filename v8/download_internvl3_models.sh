#!/bin/bash
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
