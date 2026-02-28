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
                    conf = torch.sigmoid(output_reshaped[:, 4])
                    mask = conf > conf_threshold
                    
                    if mask.sum() > 0:
                        filtered = output_reshaped[mask]
                        
                        # Convert to [x1, y1, x2, y2, conf, class]
                        for det in filtered:
                            bbox = det[:4].sigmoid() * 640  # Scale to 640
                            conf_val = torch.sigmoid(det[4])
                            
                            if det.shape[0] > 5:
                                cls_conf = torch.softmax(det[5:], dim=0)
                                cls_idx = cls_conf.argmax()
                            else:
                                cls_idx = 0
                            
                            img_preds.append([
                                bbox[0].item(), bbox[1].item(),
                                bbox[2].item(), bbox[3].item(),
                                conf_val.item(), cls_idx.item()
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
    validator = Validator(model_path=args.model, device=args.device)
    
    # Create dummy dataloader (since we don't have actual data)
    from torch.utils.data import TensorDataset
    dummy_images = torch.randn(16, 3, 640, 640)
    dummy_targets = [[] for _ in range(16)]
    
    dataset = TensorDataset(dummy_images)
    val_loader = DataLoader(dataset, batch_size=args.batch_size)
    
    # Validate
    metrics = validator.validate(val_loader)
    
    print("\nValidation Results:")
    print(f"mAP@0.5: {metrics['map_50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
