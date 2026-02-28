# experiments/test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import yaml
from pathlib import Path
import sys
import argparse
import cv2

sys.path.append('..')

from models.yolov8_enhanced import EnhancedYOLOv8
from utils.metrics import evaluate_map


class Tester:
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
        else:
            print("Warning: No model weights loaded. Using random initialization.")
        
        self.model.eval()
    
    def test_image(self, image_path, conf_threshold=0.5):
        """Run inference on a single image"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to read image: {image_path}")
            return None
        
        original_shape = img.shape
        
        # Preprocess
        img_tensor = self.preprocess(img)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Postprocess
        predictions = self.postprocess(outputs, conf_threshold)
        
        return predictions, original_shape
    
    def preprocess(self, img, input_size=640):
        """Preprocess image for inference"""
        # Resize
        h, w = img.shape[:2]
        scale = input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to square
        top = (input_size - new_h) // 2
        left = (input_size - new_w) // 2
        img_padded = cv2.copyMakeBorder(
            img_resized, top, input_size - new_h - top, left, input_size - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Normalize
        img_tensor = torch.from_numpy(img_padded).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor
    
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
                    conf = torch.sigmoid(output_reshaped[:, 4:5])
                    mask = (conf.squeeze() > conf_threshold)
                    
                    if mask.sum() > 0:
                        filtered_output = output_reshaped[mask]
                        img_preds = filtered_output.cpu().numpy().tolist()
                
                batch_preds.append(img_preds)
            
            predictions.extend(batch_preds)
        
        return predictions
    
    def test_dataset(self, test_loader):
        """Test on entire dataset"""
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc='Testing'):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test YOLOv8 model')
    parser.add_argument('--model', type=str, default='weights/best_model.pth', help='Path to model weights')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Initialize tester
    tester = Tester(model_path=args.model, config_path=args.config, device=args.device)
    
    if args.image:
        print(f"Testing on image: {args.image}")
        predictions, img_shape = tester.test_image(args.image)
        
        if predictions:
            print(f"Detected {len(predictions)} objects")
            for pred in predictions:
                print(f"  Box: {pred[:4]}, Conf: {pred[4]:.3f}")
    else:
        print("Testing on VisDrone dataset...")
        from experiments.dataset import VisDroneDataset

        test_dataset = VisDroneDataset('data/VisDrone/test/images', 'data/VisDrone/test/annotations')
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        metrics = tester.test_dataset(test_loader)
        
        print("\nTest Results:")
        print(f"mAP@0.5: {metrics.get('map_50', 0):.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('map_50_95', 0):.4f}")
