# Cloud GPU Training Guide

This project supports training on various cloud GPU platforms. Choose the one that best fits your needs.

## 🚀 Google Colab (Recommended - Free Tier Available)

**GPU:** Tesla T4 (16GB) - Free  
**Cost:** Free for up to 12 hours, then reconnect  
**Setup Time:** 5 minutes

### Quick Start:
1. Upload `colab_train.ipynb` to Google Colab
2. Upload your project folder to Google Drive at `/MyDrive/uav_small_object_detection/`
3. Runtime → Change runtime type → GPU (T4)
4. Run all cells

### Pro Tips:
- Keep browser tab open (Colab disconnects on inactivity)
- Save checkpoints to Google Drive every 5 epochs
- Free tier limited to ~12 hours per session

---

## 🔷 Kaggle Notebooks

**GPU:** Tesla P100 (16GB) or T4  
**Cost:** Free  
**Setup Time:** 5 minutes

### Quick Start:
1. Create new Kaggle Notebook
2. Settings → Accelerator → GPU
3. Add Dataset: Upload project as Kaggle Dataset
4. Install dependencies:
```python
!pip install ultralytics pyyaml tqdm wandb
```
5. Run training:
```python
!python main.py train --config configs/train_config_cloud.yaml
```

### Pro Tips:
- Free GPUs for 30 hours/week
- Persistent datasets (no re-upload)
- Automatic output saving

---

## ☁️ AWS SageMaker / EC2

**GPU:** Various (g4dn.xlarge recommended - T4)  
**Cost:** ~$0.50/hour  
**Setup Time:** 15 minutes

### EC2 Setup:
```bash
# Launch g4dn.xlarge instance with Deep Learning AMI
# SSH into instance

# Clone project
git clone <your-repo-url>
cd uav_small_object_detection

# Install dependencies (conda env already has PyTorch)
pip install -r requirements.txt

# Start training
python main.py train --config configs/train_config_cloud.yaml
```

### SageMaker Setup:
Use `sagemaker_train.py` script (create if needed) or run via Notebook instance.

---

## 🌐 Paperspace Gradient

**GPU:** Various (starting at $0.45/hour)  
**Cost:** $0.45-$3/hour  
**Setup Time:** 5 minutes

### Quick Start:
1. Create new Gradient Notebook
2. Select GPU instance (RTX 4000 recommended)
3. Terminal:
```bash
git clone <your-repo>
cd uav_small_object_detection
pip install -r requirements.txt
python main.py train --config configs/train_config_cloud.yaml
```

---

## 🎯 Lambda Labs

**GPU:** A100, A6000, RTX 6000  
**Cost:** $0.50-$1.10/hour  
**Setup Time:** 10 minutes

### Setup:
```bash
# SSH into instance
git clone <your-repo>
cd uav_small_object_detection
pip install -r requirements.txt

# Multi-GPU training (if available)
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py train --config configs/train_config_cloud.yaml
```

---

## 📊 Recommended Settings by GPU

| GPU | Batch Size | Model Scale | Expected Time (100 epochs) |
|-----|------------|-------------|----------------------------|
| T4 (16GB) | 16 | s or m | ~8-12 hours |
| V100 (16GB) | 24 | m or l | ~5-7 hours |
| A100 (40GB) | 32 | l or x | ~3-4 hours |
| RTX 4090 | 32 | l or x | ~4-5 hours |

---

## 🔧 Configuration Files

- **Local (4GB GPU):** `configs/train_config.yaml` - batch_size: 2, model: n
- **Cloud (16GB+ GPU):** `configs/train_config_cloud.yaml` - batch_size: 16, model: m

---

## 💾 Saving Checkpoints

All platforms save to `weights/best_model.pth` by default.

**For persistent storage:**
- Colab: Copy to `/content/drive/MyDrive/`
- Kaggle: Outputs auto-saved
- AWS: Use S3 bucket
- Others: Mount external storage or download via SCP

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller `model_scale` (n < s < m < l < x)
- Increase `gradient_accumulation_steps`

### Slow Training
- Verify GPU is active: `nvidia-smi`
- Check `mixed_precision: true` in config
- Use `num_workers > 0` in dataloader (not on Windows)

### Dataset Not Found
- Verify VisDrone structure: `data/VisDrone/{train,val,test}/{images,annotations}/`
- Check paths in `data/VisDrone.yaml`

---

## 📈 Monitoring Training

### Option 1: Weights & Biases (Recommended)
```yaml
# In train_config_cloud.yaml
use_wandb: true
```
Then login: `wandb login`

### Option 2: TensorBoard
```bash
tensorboard --logdir=results/
```

### Option 3: Terminal Output
Training prints loss/metrics every batch and validates every 5 epochs.

---

## 🎓 Cost Comparison (100 epochs, medium model)

| Platform | GPU | Time | Cost |
|----------|-----|------|------|
| Google Colab Free | T4 | 10h | $0 |
| Kaggle | P100 | 8h | $0 |
| Paperspace | RTX 4000 | 7h | ~$3 |
| Lambda A100 | A100 | 3h | ~$3 |
| AWS g4dn.xlarge | T4 | 10h | ~$5 |

**Recommendation:** Start with Colab/Kaggle (free), then upgrade to paid if you need faster iteration.
