# experiments/validate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import argparse
from experiments.dataset import VisDroneDataset

sys.path.append('..')

from models.yolov8_enhanced import EnhancedYOLOv8
from utils.metrics import evaluate_map


class Validator:
    def __init__(self, model_path=None, config_path='configs/train_config.yaml', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except:
            # Default config if not found
            self.config = {
                'num_classes': 10,
                'model_scale': 'l'
            }
        
        # Initialize model
        self.model = EnhancedYOLOv8(
            nc=self.config.get('num_classes', 10),
            scales=self.config.get('model_scale', 'l')
        ).to(self.device)
        
        # Load weights if provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {model_path}")
        
        self.model.eval()
    
    def validate(self, val_loader):
        """Validation loop"""
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Process predictions
                preds = self.postprocess(outputs)
                predictions.extend(preds)
                ground_truths.extend(targets)
        
        # Calculate mAP metrics
        metrics = evaluate_map(predictions, ground_truths)
        
        return metrics
    
    def postprocess(self, outputs, conf_threshold=0.5, iou_threshold=0.45):
        """Post-process model outputs"""
        predictions = []
        
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        for output in outputs:
            if output is None:
                continue
            
            batch_preds = []
            
            batch_size = output.shape[0]
            for b in range(batch_size):
                img_preds = []
                
                # Reshape output [batch, num_anchors, h, w, 5+num_classes]
                if output.dim() == 5:
                    output_reshaped = output[b].view(-1, output.shape[-1])
                else:
                    output_reshaped = output[b].view(-1, output.shape[-1])
                
                # Filter by confidence
                if output_reshaped.shape[1] > 4:
                    obj_conf = torch.sigmoid(output_reshaped[:, 4])
                    mask = obj_conf > conf_threshold
                    
                    if mask.sum() > 0:
                        filtered = output_reshaped[mask]
                        
                        # Convert to [x1, y1, x2, y2, conf, class]
                        for det in filtered:
                            bbox = torch.sigmoid(det[:4]).clamp(0.0, 1.0)
                            x1, y1, x2, y2 = bbox.tolist()
                            if x2 < x1:
                                x1, x2 = x2, x1
                            if y2 < y1:
                                y1, y2 = y2, y1
                            conf_val = torch.sigmoid(det[4])
                            
                            if det.shape[0] > 5:
                                cls_conf = torch.softmax(det[5:], dim=0)
                                cls_idx = cls_conf.argmax()
                                final_conf = conf_val * cls_conf[cls_idx]
                            else:
                                cls_idx = 0
                                final_conf = conf_val
                            
                            img_preds.append([
                                x1, y1,
                                x2, y2,
                                final_conf.item(), int(cls_idx.item())
                            ])
                
                batch_preds.append(img_preds)
            
            predictions.extend(batch_preds)
        
        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='weights/best_model.pth', help='Model path')
    parser.add_argument('--data', type=str, default='data/VisDrone.yaml', help='Data config path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    # Initialize validator
    validator = Validator(model_path=args.model, config_path='configs/train_config.yaml', device=args.device)

    # Build real VisDrone validation loader
    data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
    img_size = validator.config.get('img_size', 640)
    num_workers = validator.config.get('num_workers', 0)
    pin_memory = bool(validator.config.get('pin_memory', False) and torch.cuda.is_available())

    val_images = data_root / 'VisDrone' / 'val' / 'images'
    val_annotations = data_root / 'VisDrone' / 'val' / 'annotations'

    if not val_images.exists() or not val_annotations.exists():
        raise FileNotFoundError(
            f"Validation dataset not found. Expected: {val_images} and {val_annotations}"
        )

    val_dataset = VisDroneDataset(str(val_images), str(val_annotations), img_size=img_size)

    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        return images, list(targets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    if len(val_loader) == 0:
        raise RuntimeError("Validation loader is empty. Check dataset files under data/VisDrone/val.")

    # Validate
    metrics = validator.validate(val_loader)
    
    print("\nValidation Results:")
    print(f"mAP@0.5: {metrics['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
