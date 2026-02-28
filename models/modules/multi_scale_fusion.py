# models/modules/multi_scale_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusionModule(nn.Module):
    """Multi-scale feature fusion for improved detection"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels if isinstance(channels, (list, tuple)) else [channels]
        self.lateral_convs = nn.ModuleDict()
        self.fusion_convs = nn.ModuleDict()
    
    def forward(self, features):
        """Forward pass through multi-scale fusion"""
        if not isinstance(features, (list, tuple)):
            features = [features]
        
        if len(features) == 0:
            return features
        
        # Determine target channels (use last feature's channels)
        target_channels = features[-1].shape[1]
        
        # Apply lateral convolutions to normalize channel dimensions
        lateral_features = []
        for i, feat in enumerate(features):
            key = str(i)
            
            # Create lateral conv if needed
            if key not in self.lateral_convs:
                if feat.shape[1] == target_channels:
                    self.lateral_convs[key] = nn.Identity()
                else:
                    self.lateral_convs[key] = nn.Conv2d(
                        feat.shape[1], target_channels, kernel_size=1
                    ).to(feat.device)
            
            # Create fusion conv if needed
            if key not in self.fusion_convs:
                self.fusion_convs[key] = nn.Sequential(
                    nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(target_channels),
                    nn.ReLU(inplace=True)
                ).to(feat.device)
            
            # Apply lateral conv
            lateral_conv = self.lateral_convs[key]
            lateral_features.append(lateral_conv(feat))
        
        # Top-down pathway to fuse features
        fused_features = [lateral_features[-1]]
        
        for i in range(len(lateral_features) - 2, -1, -1):
            # Upsample the previous level
            upsampled = F.interpolate(
                fused_features[-1], 
                size=lateral_features[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # Add lateral feature
            fused = lateral_features[i] + upsampled
            fused_features.append(fused)
        
        # Reverse to match original order
        fused_features = fused_features[::-1]
        
        # Apply fusion convolutions
        output_features = []
        for i, feat in enumerate(fused_features):
            key = str(i)
            if key in self.fusion_convs:
                output_features.append(self.fusion_convs[key](feat))
            else:
                output_features.append(feat)
        
        return output_features
