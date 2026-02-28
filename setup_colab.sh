#!/bin/bash
# Colab Setup Script
echo "Setting up UAV Small Object Detection on Colab..."

# Install dependencies
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q ultralytics pyyaml tqdm numpy opencv-python pillow matplotlib scipy

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"}')"

echo "Setup complete! Run: python main.py train --config configs/train_config_cloud.yaml"
