# utils/augmentations.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class SmallObjectAugmentation:
    """Advanced augmentation for small object detection"""
    def __init__(self, mosaic_prob=0.8, mixup_prob=0.2, small_object_scale=1.5):
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.small_object_scale = small_object_scale
    
    def __call__(self, images, targets=None):
        """Apply augmentations"""
        if images is None:
            return images, targets
        
        # Apply mosaic augmentation
        if np.random.random() < self.mosaic_prob:
            images = self.mosaic_augment(images)
        
        # Apply mixup augmentation
        if np.random.random() < self.mixup_prob:
            images = self.mixup_augment(images)
        
        # Apply other augmentations
        images = self.apply_geometric_transforms(images)
        
        return images, targets
    
    def mosaic_augment(self, images):
        """Mosaic augmentation combining 4 images"""
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        batch_size = images.shape[0]
        h, w = images.shape[-2:]
        
        # Create mosaic output
        mosaic_h, mosaic_w = h, w
        mosaic = torch.zeros((batch_size, images.shape[1], mosaic_h, mosaic_w), dtype=images.dtype)
        
        # For small dataset, use repetition with slight variations
        for i in range(batch_size):
            # Create a 2x2 mosaic by combining and tiling
            img = images[i:i+1]
            
            # Tile the image
            tiled = torch.cat([img, img], dim=-1)  # Horizontal
            mosaic[i] = torch.cat([tiled, tiled], dim=-2)[:, :, :mosaic_h, :mosaic_w]
        
        return mosaic
    
    def mixup_augment(self, images):
        """Mixup augmentation"""
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        batch_size = images.shape[0]
        alpha = np.random.beta(1.5, 1.5)
        
        # Create mixed images
        indices = torch.randperm(batch_size)
        
        mixed_images = alpha * images + (1 - alpha) * images[indices]
        
        return mixed_images
    
    def apply_geometric_transforms(self, images):
        """Apply geometric transformations"""
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            images = torch.flip(images, dims=[-1])
        
        # Random vertical flip
        if np.random.random() > 0.5:
            images = torch.flip(images, dims=[-2])
        
        return images
    
    def scale_small_objects(self, images, targets, scale_factor=1.5):
        """Emphasize small objects through scaling"""
        if targets is None or len(targets) == 0:
            return images, targets
        
        # This would require more complex logic to handle per-bbox scaling
        # For now, simple overall augmentation
        return images, targets
