import os
import torch
import cv2
from torch.utils.data import Dataset

class VisDroneDataset(Dataset):
    """VisDrone dataset loader for small object detection"""
    def __init__(self, img_dir, ann_dir, img_size=640):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_size = img_size
        
        # Get list of images
        self.img_files = []
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.img_files.append(os.path.join(img_dir, f))
        
        if len(self.img_files) == 0:
            print(f"Warning: No images found in {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            # Load image
            img_path = self.img_files[idx]
            img = cv2.imread(img_path)
            
            if img is None:
                # Return dummy data if image can't be loaded
                return torch.zeros((3, self.img_size, self.img_size)), []
            
            # Resize image
            h, w = img.shape[:2]
            scale = self.img_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            
            # Pad to square
            top = (self.img_size - new_h) // 2
            left = (self.img_size - new_w) // 2
            img = cv2.copyMakeBorder(
                img, top, self.img_size - new_h - top, 
                left, self.img_size - new_w - left,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            
            # Convert to tensor and normalize
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC -> CHW
            
            # Load annotations
            ann_path = os.path.join(
                self.ann_dir, 
                os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            )
            
            targets = []
            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        # Expect VisDrone format: x,y,w,h,score,category,truncation,occlusion
                        if len(parts) >= 8:
                            try:
                                x, y, w, h = [float(p) for p in parts[:4]]
                                score = float(parts[4])
                                category = int(float(parts[5]))
                                truncation = float(parts[6])
                                occlusion = float(parts[7])

                                # Skip ignored category (0) or heavily occluded
                                if category == 0 or occlusion > 2:
                                    continue

                                # Map VisDrone category IDs (1-10) to 0-based class ids (0-9)
                                class_id = category - 1
                                if class_id < 0 or class_id >= 10:
                                    continue

                                # Convert to normalized coordinates
                                x1 = max(0, (x * scale + left) / self.img_size)
                                y1 = max(0, (y * scale + top) / self.img_size)
                                x2 = min(1, ((x + w) * scale + left) / self.img_size)
                                y2 = min(1, ((y + h) * scale + top) / self.img_size)

                                targets.append([x1, y1, x2, y2, class_id])
                            except Exception:
                                continue
            
            return img, targets
        except Exception as e:
            print(f"Error loading {idx}: {e}")
            return torch.zeros((3, self.img_size, self.img_size)), []
