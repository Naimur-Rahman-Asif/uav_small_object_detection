# models/modules/deformable_conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv2d(nn.Module):
    """Deformable convolution module for adaptive spatial sampling"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Regular convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
        # Offset generation network
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, stride, padding, groups=groups)
        )
        
        # Initialize offset to zero
        self.offset_conv[-1].weight.data.zero_()
        self.offset_conv[-1].bias.data.zero_()
    
    def forward(self, x):
        # Generate offsets
        offsets = self.offset_conv(x)
        
        # Apply deformable convolution using regular conv as approximation
        # This is a simplified version that applies regular convolution
        # A full implementation would sample features at offset positions
        x_out = self.conv(x)
        
        # Apply learned offset as a modulation factor
        offset_scale = torch.sigmoid(offsets)
        batch_size = x.size(0)
        offset_scale = offset_scale.view(batch_size, -1, x.size(2), x.size(3))
        
        # Modulate output features
        x_out = x_out * torch.mean(offset_scale, dim=1, keepdim=True)
        
        return x_out
