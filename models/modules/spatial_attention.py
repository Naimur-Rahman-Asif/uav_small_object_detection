# models/modules/spatial_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidAttention(nn.Module):
    """Spatial pyramid attention module for small object detection"""
    def __init__(self, channels, reduction=16, pyramid_levels=3):
        super().__init__()
        self.channels = channels
        self.pyramid_levels = pyramid_levels
        self.reduction = reduction
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention with pyramid structure
        self.spatial_attentions = nn.ModuleList()
        for _ in range(pyramid_levels):
            self.spatial_attentions.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels // reduction, 1, kernel_size=1),
                    nn.Sigmoid()
                )
            )
    
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial pyramid attention
        spatial_atts = []
        for i, spatial_att_module in enumerate(self.spatial_attentions):
            # Extract features at different scales
            scale_factor = 2 ** i
            if scale_factor > 1:
                x_scaled = F.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor)
            else:
                x_scaled = x
            
            # Apply spatial attention
            spatial_att = spatial_att_module(x_scaled)
            
            # Restore original resolution
            if scale_factor > 1:
                spatial_att = F.interpolate(
                    spatial_att, 
                    size=x.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            spatial_atts.append(spatial_att)
        
        # Combine spatial attention maps
        combined_spatial_att = torch.stack(spatial_atts, dim=0).mean(dim=0)
        x = x * combined_spatial_att
        
        return x
