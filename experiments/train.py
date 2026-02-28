# experiments/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import wandb
import yaml
from pathlib import Path
import sys
import cv2
import os

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from models.yolov8_enhanced import EnhancedYOLOv8
from utils.losses import EnhancedLoss
from utils.augmentations import SmallObjectAugmentation
from utils.metrics import evaluate_map
from experiments.dataset import VisDroneDataset

class UAVTrainer:
    def __init__(self, config_path='configs/train_config.yaml'):
        # Handle path for both root and experiments directory execution
        config_file = Path(config_path)
        if not config_file.exists():
            config_file = Path('..') / config_path
            if not config_file.exists():
                config_file = Path(config_path)  # Try original path
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory optimization for small GPUs (MX250 has 4GB)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Enable expandable segments to reduce fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Initialize model
        self.model = EnhancedYOLOv8(
            nc=self.config['num_classes'],
            scales=self.config['model_scale']
        ).to(self.device)
        
        # Custom loss function for small objects
        self.criterion = EnhancedLoss(
            nc=self.config['num_classes'],
            device=self.device
        )
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['warmup_epochs'],
            T_mult=2
        )
        
        # Gradient scaler for mixed precision
        # Use new torch.amp API to avoid deprecation warning
        try:
            self.scaler = torch.amp.GradScaler(enabled=self.config['mixed_precision'])
        except (AttributeError, TypeError):
            # Fallback for older PyTorch versions
            self.scaler = amp.GradScaler(enabled=self.config['mixed_precision'])
        
        # Data augmentation
        self.augmentation = SmallObjectAugmentation(
            mosaic_prob=0.8,
            mixup_prob=0.2,
            small_object_scale=1.5  # Enhanced scaling for small objects
        )
        
        # Initialize wandb for experiment tracking
        if self.config.get('use_wandb', False):
            try:
                wandb.init(
                    project="uav-small-object-detection",
                    config=self.config,
                    name=f"YOLOv8-SAD_{self.config['experiment_name']}"
                )
                self.use_wandb = True
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                print("Continuing without experiment tracking...")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def train_epoch(self, train_loader, epoch):
        """Training loop for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        accum_steps = self.config.get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Handle list targets - convert to tensors if needed
            if isinstance(targets, list) and len(targets) > 0:
                # targets is a list of lists, keep as is for the loss function
                targets = targets
            
            # Mixed precision training
            try:
                # Use new torch.amp API to avoid deprecation warning
                with torch.amp.autocast('cuda', enabled=self.config['mixed_precision']):
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)
                    loss = loss / accum_steps  # Scale loss for gradient accumulation
            except (AttributeError, Exception):
                # Fallback for older PyTorch versions
                with amp.autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, targets)
                    loss = loss / accum_steps
            
            # Backward pass (accumulate gradients)
            self.scaler.scale(loss).backward()
            
            # Optimizer step every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item() * accum_steps  # Unscale for display
            progress_bar.set_postfix({
                'loss': loss.item() * accum_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                try:
                    wandb.log({
                        'batch_loss': loss.item() * accum_steps,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                except Exception as e:
                    print(f"Warning: Could not log to wandb: {e}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Process predictions
                preds = self.postprocess(outputs)
                predictions.extend(preds)
                ground_truths.extend(targets)
        
        # Calculate mAP metrics
        metrics = evaluate_map(predictions, ground_truths)
        
        # Log detailed small object metrics
        small_obj_metrics = self.evaluate_small_objects(predictions, ground_truths)
        metrics.update(small_obj_metrics)
        
        return metrics
    
    def evaluate_small_objects(self, predictions, ground_truths, threshold=32):
        """Specialized evaluation for small objects"""
        small_preds = []
        small_gts = []
        
        for preds, gts in zip(predictions, ground_truths):
            # Filter small objects (based on pixel area)
            small_pred = [p for p in preds if (p[3] * p[4]) * 640 * 640 < threshold * threshold]
            small_gt = [g for g in gts if (g[3] * g[4]) * 640 * 640 < threshold * threshold]
            
            small_preds.append(small_pred)
            small_gts.append(small_gt)
        
        # Calculate small object mAP
        small_map = evaluate_map(small_preds, small_gts)
        
        return {
            'small_map_50': small_map['map_50'],
            'small_map_50_95': small_map['map_50_95'],
            'small_precision': small_map['precision'],
            'small_recall': small_map['recall']
        }
    
    def train(self, train_loader, val_loader):
        """Complete training pipeline"""
        best_map = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            if epoch % self.config['val_interval'] == 0:
                metrics = self.validate(val_loader)
                
                # Save best model
                if metrics['map_50_95'] > best_map:
                    best_map = metrics['map_50_95']
                    
                    # Ensure weights directory exists
                    weights_dir = Path('..') / 'weights' if not Path('weights').exists() else Path('weights')
                    weights_dir.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_map': best_map,
                        'config': self.config
                    }, weights_dir / 'best_model.pth')
                
                # Log metrics
                print(f"\nEpoch {epoch} Metrics:")
                print(f"mAP@0.5: {metrics['map_50']:.4f}")
                print(f"mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
                print(f"Small Object mAP: {metrics['small_map_50_95']:.4f}")
                
                if self.use_wandb:
                    try:
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_loss,
                            **metrics
                        })
                    except Exception as e:
                        print(f"Warning: Could not log to wandb: {e}")
            
            # Learning rate scheduling
            self.scheduler.step()
        
        print(f"Training completed. Best mAP: {best_map:.4f}")

if __name__ == '__main__':
    # Configuration
    config = {
        'experiment_name': 'visdrone_small_objects',
        'num_classes': 10,
        'model_scale': 'l',
        'epochs': 300,
        'batch_size': 16,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'warmup_epochs': 10,
        'mixed_precision': True,
        'grad_clip': 10.0,
        'val_interval': 5,
        'use_wandb': False  # Disable wandb by default to avoid auth issues
    }
    
    # Save config (handle path for both root and experiments directory execution)
    config_path = Path('configs/train_config.yaml')
    if not config_path.exists():
        config_path = Path('..') / 'configs' / 'train_config.yaml'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize trainer
    trainer = UAVTrainer(str(config_path))
    
    # Load datasets
    
    # Create datasets with proper paths
    data_root = Path('..') / 'data' if not Path('data').exists() else Path('data')
    
    train_dataset = VisDroneDataset(
        str(data_root / 'VisDrone' / 'train' / 'images'),
        str(data_root / 'VisDrone' / 'train' / 'annotations')
    )
    val_dataset = VisDroneDataset(
        str(data_root / 'VisDrone' / 'val' / 'images'),
        str(data_root / 'VisDrone' / 'val' / 'annotations')
    )
    
    # Custom collate function to handle variable-length targets
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        # Keep targets as list of lists since they have variable length
        return images, list(targets)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Set to 0 on Windows
        pin_memory=False,  # Set to False since we're not on GPU
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(1, config['batch_size'] // 2), 
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Start training
    if len(train_loader) > 0:
        trainer.train(train_loader, val_loader)
    else:
        print("Error: No data available. Please download the dataset first.")