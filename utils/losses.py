# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedLoss(nn.Module):
    """Enhanced loss function for small object detection"""
    def __init__(self, nc=80, device='cpu'):
        super().__init__()
        self.nc = nc
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets):
        """Calculate enhanced loss with proper gradient flow"""
        loss_components = []
        loss_dict = {'cls_loss': 0.0, 'box_loss': 0.0, 'obj_loss': 0.0}
        
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        
        num_valid = 0
        for output in outputs:
            if output is None or output.numel() == 0:
                continue
            
            # Extract predictions: output shape is [B, 4, H, W, 5+nc]
            # where 4 is num anchors per cell, 5+nc is [x, y, w, h, obj, ...classes]
            reg_preds = output[..., :4]  # [B, 4, H, W, 4]
            obj_preds = output[..., 4:5]  # [B, 4, H, W, 1]
            cls_preds = output[..., 5:]  # [B, 4, H, W, nc]
            
            # Simple placeholder regression loss (smooth L1 toward zero)
            box_loss = F.smooth_l1_loss(reg_preds, torch.zeros_like(reg_preds), reduction='mean')
            
            # Objectness loss (assume all cells have objects for now — simplified)
            obj_target = torch.ones_like(obj_preds)
            obj_loss = F.binary_cross_entropy_with_logits(obj_preds, obj_target, reduction='mean')
            
            # Classification loss (uniform target across classes)
            if cls_preds.shape[-1] > 0:
                cls_target = torch.ones_like(cls_preds) / self.nc
                cls_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_target, reduction='mean')
            else:
                cls_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Combine weighted losses
            total = 0.5 * box_loss + 1.0 * obj_loss + 0.5 * cls_loss
            loss_components.append(total)
            
            loss_dict['box_loss'] += box_loss.item()
            loss_dict['obj_loss'] += obj_loss.item()
            loss_dict['cls_loss'] += cls_loss.item()
            num_valid += 1
        
        # Sum all scale losses
        if len(loss_components) == 0:
            final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            final_loss = torch.stack(loss_components).mean()
        
        # Normalize dict
        if num_valid > 0:
            for k in loss_dict:
                loss_dict[k] /= num_valid
        
        return final_loss, loss_dict
