# utils/metrics.py
import numpy as np
import torch
from typing import List, Dict, Tuple


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)
    
    # Ensure we have valid boxes
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    
    # Ensure coordinates are in correct format
    if box1[2] < box1[0]:
        box1[0], box1[2] = box1[2], box1[0]
    if box1[3] < box1[1]:
        box1[1], box1[3] = box1[3], box1[1]
    if box2[2] < box2[0]:
        box2[0], box2[2] = box2[2], box2[0]
    if box2[3] < box2[1]:
        box2[1], box2[3] = box2[3], box2[1]
    
    # Calculate area of intersection
    x_left = max(box1[0], box2[0])
    x_right = min(box1[2], box2[2])
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def evaluate_map(predictions, ground_truths, iou_threshold=0.5, num_classes=None):
    """
    Evaluate Mean Average Precision (mAP) metric
    
    Args:
        predictions: List of predicted bounding boxes [x1, y1, x2, y2, conf, class]
        ground_truths: List of ground truth bounding boxes [x1, y1, x2, y2, class]
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        Dictionary with mAP metrics
    """
    if not predictions or not ground_truths:
        return {
            'map_50': 0.0,
            'map_50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truths, torch.Tensor):
        ground_truths = ground_truths.cpu().numpy()
    
    # Handle list of lists/tensors
    if predictions and isinstance(predictions[0], (list, tuple)):
        predictions = [item for sublist in predictions for item in sublist]
    if ground_truths and isinstance(ground_truths[0], (list, tuple)):
        ground_truths = [item for sublist in ground_truths for item in sublist]
    
    # Convert to array
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions) if predictions else np.array([])
    if not isinstance(ground_truths, np.ndarray):
        ground_truths = np.array(ground_truths) if ground_truths else np.array([])
    
    # Handle empty cases
    if predictions.size == 0 or ground_truths.size == 0:
        return {
            'map_50': 0.0,
            'map_50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    # Ensure predictions have at least 5 elements: [x1, y1, x2, y2, conf]
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if predictions.shape[1] < 5:
        # Pad predictions if needed
        num_boxes = predictions.shape[0]
        padded = np.ones((num_boxes, 6))
        padded[:, :predictions.shape[1]] = predictions
        predictions = padded
    
    # Ensure ground truths have format [x1, y1, x2, y2, ...]
    if ground_truths.ndim == 1:
        ground_truths = ground_truths.reshape(1, -1)
    
    # Calculate IoU between all predictions and ground truths
    if len(predictions) > 0 and len(ground_truths) > 0:
        tp = 0
        fp = 0
        
        # For each prediction, find best matching ground truth
        matched_gt = set()
        
        # Sort by confidence
        if predictions.shape[1] >= 5:
            sorted_indices = np.argsort(-predictions[:, 4])  # Sort by confidence
            predictions = predictions[sorted_indices]
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                iou = compute_iou(pred[:4], gt[:4])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(ground_truths) - len(matched_gt)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Approximate mAP (simplified)
        map_50_95 = precision * recall
        map_50 = precision * recall
    else:
        tp = fp = fn = 0
        precision = recall = map_50_95 = map_50 = 0.0
    
    return {
        'map_50': float(map_50),
        'map_50_95': float(map_50_95),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_F1(precision, recall):
    """Compute F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


class MetricsCalculator:
    """Advanced metrics calculation for object detection"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
    
    def update(self, predictions, ground_truths, iou_threshold=0.5):
        """Update metrics with batch"""
        if len(predictions) == 0 or len(ground_truths) == 0:
            return
        
        for pred, gt in zip(predictions, ground_truths):
            if len(pred) == 0:
                self.fn += np.ones(self.num_classes)
                continue
            
            matched_gt = set()
            
            for p in pred:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, g in enumerate(gt):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = compute_iou(p[:4], g[:4])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                pred_class = int(p[-1]) if len(p) > 5 else 0
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    self.tp[pred_class] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    self.fp[pred_class] += 1
            
            # Count false negatives
            for gt_idx in range(len(gt)):
                if gt_idx not in matched_gt:
                    gt_class = int(gt[gt_idx][-1]) if len(gt[gt_idx]) > 4 else 0
                    self.fn[gt_class] += 1
    
    def compute_metrics(self):
        """Compute precision, recall, and mAP"""
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        map_score = np.mean(precision * recall)
        
        return {
            'precision': precision.mean(),
            'recall': recall.mean(),
            'f1': f1.mean(),
            'mAP': map_score,
            'precision_per_class': precision,
            'recall_per_class': recall
        }
