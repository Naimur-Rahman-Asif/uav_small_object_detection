# 🚀 Cloud GPU Training - Quick Start

## Fastest Path: Google Colab (5 minutes)

### Option A: Upload Project from Local

1. **Find the package:**
   ```
   ✓ uav_detection_cloud.zip (already created in your project folder)
   ```

2. **Upload to Google Drive:**
   - Go to [Google Drive](https://drive.google.com)
   - Create folder: `uav_small_object_detection`
   - Upload and extract `uav_detection_cloud.zip` there
   - Upload your VisDrone dataset to:
     - `data/VisDrone/train/images/` + `annotations/`
     - `data/VisDrone/val/images/` + `annotations/`
     - `data/VisDrone/test/images/` + `annotations/`

3. **Open Colab Notebook:**
   - In Google Drive, right-click `colab_train.ipynb`
   - Select "Open with" → "Google Colaboratory"
   - Runtime → Change runtime type → **T4 GPU**
   - Run all cells (Ctrl+F9)

4. **Training starts automatically!**
   - Progress bars show training status
   - Results saved to Google Drive
   - ~8-10 hours for 100 epochs on T4

---

### Option B: Direct Upload to Colab

1. **Open Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook → Select `colab_train.ipynb` from your local project

2. **Upload project files:**
   - Run the "Upload files" cell in the notebook
   - Upload `uav_detection_cloud.zip`
   - Extract it in Colab

3. **Upload dataset:**
   - Upload VisDrone dataset ZIP (or use Drive mounting)
   - Extract to correct folders

4. **Select GPU & Run:**
   - Runtime → Change runtime type → **T4 GPU**
   - Run all cells

---

## Configuration Comparison

| Setting | Local (4GB) | Cloud (T4 16GB) | Speed Difference |
|---------|-------------|-----------------|------------------|
| Model | nano (n) | medium (m) | 3x larger |
| Batch Size | 2 | 16 | 8x |
| Training Time | ~100 hours | ~8 hours | **12x faster** |
| Cost | $0 | $0 (free tier) | - |

---

## Cloud Commands Reference

### Once in Colab/Cloud environment:

```bash
# Verify GPU
!nvidia-smi

# Install dependencies (if not using notebook)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install ultralytics pyyaml tqdm opencv-python

# Train with cloud config
!python main.py train --config configs/train_config_cloud.yaml --device cuda

# Test trained model
!python main.py validate --model weights/best_model.pth --config configs/train_config_cloud.yaml

# Inference on image
!python main.py infer --model weights/best_model.pth --image data/VisDrone/test/images/0000006_00159_d_0000001.jpg
```

---

## Monitoring Training

### Progress Display:
```
Epoch 0: 15%|████▌          | 242/1617 [01:30<08:32, 2.68it/s, loss=28.4, lr=0.001]
```
- Shows current batch progress
- Loss value (lower is better)
- Learning rate
- Estimated time remaining

### Key Metrics:
- **First few epochs:** loss ~30-40 (normal)
- **After 20 epochs:** loss ~10-15
- **After 50 epochs:** loss ~5-8
- **Good convergence:** loss < 5

---

## Saving Results

### Automatic saves:
- **Best model:** `weights/best_model.pth` (lowest validation loss)
- **Latest checkpoint:** `weights/last_model.pth`
- **Logs:** `results/` folder

### Download from Colab:
```python
# In Colab notebook cell:
from google.colab import files

# Download best model
files.download('weights/best_model.pth')

# Or zip all weights
!zip -r weights.zip weights/
files.download('weights.zip')
```

### Sync to Google Drive (safer):
```python
# Copy weights to Drive (persists across sessions)
!cp -r weights /content/drive/MyDrive/uav_detection/
!cp -r results /content/drive/MyDrive/uav_detection/
```

---

## Troubleshooting

### "Not connected to a GPU"
→ Runtime → Change runtime type → Select **GPU (T4)**

### "CUDA out of memory"
→ Reduce batch_size in `configs/train_config_cloud.yaml`:
```yaml
batch_size: 8  # Instead of 16
```

### "Dataset not found"
→ Verify paths match:
```
data/VisDrone/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

### Colab disconnects
- Colab free tier: 12-hour max runtime
- Keep browser tab open
- Use Drive sync to avoid losing progress

---

## After Training

### Evaluate on Test Set:
```bash
!python main.py test --model weights/best_model.pth --config configs/train_config_cloud.yaml
```

### Expected Results (100 epochs):
- **mAP@0.5:** 0.45-0.55
- **mAP@0.5:0.95:** 0.25-0.35
- **Small object mAP:** 0.20-0.30

### Visualize Predictions:
```python
!python main.py infer --model weights/best_model.pth --image test_image.jpg
```

---

## Cost & Time Estimates

| Platform | GPU | Time (100 epochs) | Cost | Notes |
|----------|-----|-------------------|------|-------|
| **Colab Free** | T4 | 8-10 hours | $0 | 12h limit, reconnect |
| **Colab Pro** | T4/V100 | 6-8 hours | $10/mo | 24h limit |
| **Kaggle** | P100 | 7-9 hours | $0 | 30h/week |
| **Paperspace** | RTX 4000 | 5-7 hours | ~$3 | Pay per use |
| **Lambda** | A100 | 3-4 hours | ~$3 | Fastest |

**Recommendation:** Start with Colab Free, upgrade if you need more time.

---

## Need Help?

1. **Check logs:** Look at terminal output for error messages
2. **Verify GPU:** Run `!nvidia-smi` in Colab
3. **Dataset:** Ensure VisDrone structure matches exactly
4. **Config:** Compare your config with `train_config_cloud.yaml`
5. **Documentation:** See [CLOUD_SETUP.md](CLOUD_SETUP.md) for detailed guides

---

**Ready to train? → Open `colab_train.ipynb` in Google Colab and run all cells!**
