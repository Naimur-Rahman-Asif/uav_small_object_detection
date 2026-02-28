# UAV Small Object Detection - YOLOv8 Enhanced

Advanced deep learning project for detecting small objects in UAV (drone) imagery using an enhanced YOLOv8 architecture optimized for the VisDrone dataset.

## 🚀 Quick Start Options

| Platform | GPU | Setup Time | Cost | Guide |
|----------|-----|------------|------|-------|
| **Google Colab** ⭐ | T4 (16GB) | 5 min | Free | [colab_train.ipynb](colab_train.ipynb) |
| **Kaggle Notebooks** | P100/T4 | 5 min | Free | [CLOUD_SETUP.md](CLOUD_SETUP.md) |
| **Local (4GB GPU)** | MX250+ | 10 min | - | See below |
| **AWS/Cloud** | Custom | 15 min | Varies | [CLOUD_SETUP.md](CLOUD_SETUP.md) |

💡 **Recommended:** Start with Google Colab for free GPU training!

## Features

- **Enhanced YOLOv8 Architecture**: Lightweight model optimized for small objects
- **Small Object Focus**: Specialized detection head for small object optimization
- **VisDrone Dataset**: Support for 10 classes from the VisDrone UAV dataset
- **Advanced Augmentation**: Mosaic, mixup, and geometric transformations
- **Mixed Precision Training**: GPU-optimized training with automatic mixed precision
- **Experiment Tracking**: Integration with Weights & Biases (wandb)
- **Cloud & Local**: Optimized configs for both cloud GPUs and local 4GB GPUs

## Project Structure

```
├── configs/                 # Configuration files
│   ├── train_config.yaml   # Training configuration
│   └── yolov8_custom.yaml  # Model configuration
├── data/                    # Dataset directory
│   ├── VisDrone.yaml       # Dataset manifest
│   └── scripts/             # data-related scripts (downloader removed)
├── experiments/             # Training and evaluation scripts
│   ├── train.py            # Training script
│   ├── test.py             # Testing script
│   ├── validate.py         # Validation script
│   └── comparison.py       # Model comparison
├── models/                  # Model architecture
│   ├── yolov8_enhanced.py  # Main model
│   └── modules/            # Custom modules
│       ├── spatial_attention.py
│       ├── deformable_conv.py
│       └── multi_scale_fusion.py
├── utils/                   # Utility functions
│   ├── losses.py           # Custom loss functions
│   ├── metrics.py          # Evaluation metrics
│   └── augmentations.py    # Data augmentation
├── weights/                 # Pre-trained weights
├── results/                 # Training results
├── requirements.txt        # Python dependencies
├── setup.bat              # Windows setup script
├── main.py                # Entry point
└── README.md              # This file
```

## Installation

### Step 1: Clone or Download Repository

```bash
cd your/workspace
git clone <repository-url>
cd uav_small_object_detection
```

### Step 2: Run Setup Script (Windows)

```bash
setup.bat
```

This will:
- Create a Python virtual environment
- Install all required packages
- Create necessary directories

### Step 3: Manual Setup (Linux/Mac or if setup.bat fails)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir weights results
```

## Quick Start

### 1. Prepare Dataset

Place the VisDrone dataset under `data/VisDrone` with the following layout:

- `data/VisDrone/train/images` and `data/VisDrone/train/annotations`
- `data/VisDrone/val/images` and `data/VisDrone/val/annotations`
- `data/VisDrone/test/images` and `data/VisDrone/test/annotations`

If you already have the dataset downloaded, ensure it matches the structure above. The project assumes the dataset is present and will not attempt to download it.

## ☁️ Cloud GPU Training (Recommended)

**For faster training on free cloud GPUs (T4/V100), use Google Colab:**

1. **Prepare package for cloud:**
   ```bash
   python prepare_cloud.py
   ```

2. **Upload to Google Drive:**
   - Upload `uav_detection_cloud.zip` to your Google Drive
   - Extract it to `/MyDrive/uav_small_object_detection/`
   - Copy your VisDrone dataset to the `data/VisDrone/` folders

3. **Open in Colab:**
   - Right-click `colab_train.ipynb` → Open with → Google Colaboratory
   - Runtime → Change runtime type → GPU (T4)
   - Run all cells

**Training on cloud GPU:**
- **Speed:** ~2 hours for 100 epochs (vs. 100+ hours on local 4GB GPU)
- **Cost:** Free on Colab/Kaggle
- **Settings:** Automatically uses optimized config with batch_size=16, model_scale=m

📖 **Detailed cloud setup:** See [CLOUD_SETUP.md](CLOUD_SETUP.md) for AWS, Kaggle, Paperspace, etc.

---

## 💻 Local Training (4GB GPU/CPU)

### 2. Train Model (Local)

```bash
# From project root
cd experiments
python train.py

# OR from project root
python main.py train --config configs/train_config.yaml
```

**Training Configuration** (configs/train_config.yaml):
- Batch Size: 16
- Epochs: 300
- Learning Rate: 0.001
- Mixed Precision: Enabled
- Validation Interval: Every 5 epochs

### 3. Test Model

```bash
python main.py test --model weights/best_model.pth

# OR run on single image
python main.py infer --model weights/best_model.pth --image path/to/image.jpg
```

### 4. Validate Results

```bash
cd experiments
python validate.py
```

## Detailed Usage

### Training

```bash
cd experiments
python train.py
```

**Expected Output:**
- Training progress with loss metrics
- Validation every 5 epochs
- Best model saved to `weights/best_model.pth`
- Experiment tracking in Weights & Biases (if enabled)

### Inference on Custom Image

```bash
python main.py infer --model weights/best_model.pth --image my_image.jpg
```

**Output:**
- Detected objects with confidence scores
- Bounding box coordinates

### Using the API

```python
from experiments.test import Tester
from pathlib import Path

# Initialize
tester = Tester(model_path='weights/best_model.pth', device='cuda')

# Run inference
predictions, img_shape = tester.test_image('image.jpg')

# Process results
for bbox, conf, cls in predictions:
    print(f"Box: {bbox}, Confidence: {conf:.3f}, Class: {cls}")
```

## Configuration

### Train Config (configs/train_config.yaml)

```yaml
batch_size: 16           # Batch size for training
epochs: 300              # Number of training epochs
experiment_name: visdrone_small_objects  # Experiment name
grad_clip: 10.0          # Gradient clipping value
lr: 0.001                # Learning rate
mixed_precision: true    # Enable mixed precision training
model_scale: l           # Model scale (n/s/m/l/x)
num_classes: 10          # Number of VisDrone classes
use_wandb: true          # Use Weights & Biases
val_interval: 5          # Validation interval (epochs)
warmup_epochs: 10        # Warmup epochs
weight_decay: 0.0005     # Weight decay
```

### Model Scales

- **n (nano)**: ~3M parameters, fast inference
- **s (small)**: ~11M parameters
- **m (medium)**: ~25M parameters
- **l (large)**: ~43M parameters (default)
- **x (xlarge)**: ~68M parameters, best accuracy

## Dataset: VisDrone

The VisDrone dataset contains 10 object classes:

1. Pedestrian
2. People
3. Bicycle
4. Car
5. Van
6. Truck
7. Tricycle
8. Awning-tricycle
9. Bus
10. Motor

**Dataset Size:**
- Training: ~6,471 images
- Validation: ~548 images
- Test: ~1,610 images

## Advanced Features

### Custom Loss Function

The model uses an enhanced loss function optimized for small objects:

```python
from utils.losses import EnhancedLoss

loss_fn = EnhancedLoss(nc=10, device='cuda')
outputs = model(images)
loss, loss_dict = loss_fn(outputs, targets)
```

### Data Augmentation

Enhanced augmentation for small objects:

```python
from utils.augmentations import SmallObjectAugmentation

augmenter = SmallObjectAugmentation(
    mosaic_prob=0.8,
    mixup_prob=0.2,
    small_object_scale=1.5
)
```

### Metrics Evaluation

```python
from utils.metrics import evaluate_map, MetricsCalculator

metrics = evaluate_map(predictions, ground_truths, iou_threshold=0.5)
print(f"mAP@0.5: {metrics['map_50']:.4f}")
print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
```

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in config or use CPU
```bash
python main.py train --device cpu
# OR reduce batch_size in configs/train_config.yaml
```

### Issue: "No data available"
**Solution**: Download the dataset first
```bash
python main.py download
```

### Issue: Import errors
**Solution**: Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Slow training on Windows
**Solution**: Reduce num_workers in DataLoader (already set to 0 by default)

## Performance Expectations

**Typical Results on VisDrone**:
- mAP@0.5: 35-45%
- mAP@0.5:0.95: 20-30%
- Small Object mAP: 15-25%
- Inference Speed: 30-50 FPS (on RTX 3080)

## Model Architecture

### Backbone
- Enhanced CSPDarknet with deformable convolutions
- High-resolution branch for small object features
- Dilated convolutions for larger receptive fields

### Neck (FPN)
- Multi-scale feature pyramid
- Lateral connections
- Top-down pathway fusion

### Head
- Specialized small object detection head
- 4x anchor multiplier for dense objects
- Classification and regression paths
- Context aggregation module

## Citation

```bibtex
@inproceedings{VisDrone2021,
  title = {VisDrone-DET2021: The Vision Meets Drone Object Detection Challenge Results},
  author = {Zhu, Pengfei and others},
  booktitle = {ICCV},
  year = {2021}
}
```

## License

[Your License Here]

## Support

For issues, questions, or contributions:
- Check the troubleshooting section
- Review configuration settings
- Ensure dataset is properly downloaded
- Check GPU memory availability

## Future Improvements

- [ ] Multi-GPU training support
- [ ] TensorRT optimization
- [ ] ONNX export
- [ ] Real-time video inference
- [ ] Post-training quantization
- [ ] Knowledge distillation
- [ ] Additional dataset support

---

**Last Updated**: February 2026
**Python Version**: 3.9+
**PyTorch Version**: 2.0+
