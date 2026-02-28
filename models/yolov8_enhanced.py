# models/yolov8_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import math
from typing import List, Tuple, Optional

# Enhanced Modules
from models.modules.spatial_attention import SpatialPyramidAttention
from models.modules.deformable_conv import DeformableConv2d
from models.modules.multi_scale_fusion import MultiScaleFusionModule

class EnhancedConv(nn.Module):
    """Lightweight convolution block for memory-constrained GPUs"""
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
        
        # Residual connection - only use if channels match and stride is 1
        self.use_shortcut = (c1 == c2 and s == 1)
        
    def forward(self, x):
        identity = x if self.use_shortcut else None
        out = self.act(self.bn(self.conv(x)))
        if self.use_shortcut:
            return out + identity
        return out

class SmallObjectDetectionHead(nn.Module):
    """Specialized head for small object detection"""
    def __init__(self, nc=80, channels=(256, 512, 1024)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(channels)  # number of detection layers
        self.channels = channels
        
        # Enhanced detection heads with increased capacity for small objects
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        
        for i in range(self.nl):
            # Increased channel dimensions for small object features
            ch = channels[i]
            
            # Classification path
            cls_conv = nn.Sequential(
                EnhancedConv(ch, ch, 3),
                EnhancedConv(ch, ch, 3),
                nn.Conv2d(ch, nc * 4, 1)  # 4x more anchors for small objects
            )
            self.cls_convs.append(cls_conv)
            
            # Regression path
            reg_conv = nn.Sequential(
                EnhancedConv(ch, ch, 3),
                EnhancedConv(ch, ch, 3),
                nn.Conv2d(ch, 4 * 4, 1)  # 4 anchors per grid
            )
            self.reg_convs.append(reg_conv)
        
        # Feature pyramid enhancement
        self.fpn_fusion = MultiScaleFusionModule(channels)
        
        # Context aggregation module for deepest features
        if len(channels) > 0:
            self.context_module = ContextAggregationModule(channels[-1])
        
    def forward(self, features):
        """Forward pass through detection head"""
        if not isinstance(features, (list, tuple)):
            features = [features]
        
        if len(features) == 0:
            return []
        
        # Convert to list to allow modification
        features = list(features)
        
        # Context aggregation on deepest features
        if hasattr(self, 'context_module') and len(features) > 0:
            features[-1] = self.context_module(features[-1])
        
        outputs = []
        num_scales = min(len(features), self.nl)
        for i in range(num_scales):
            feat = features[i]
            expected_ch = self.channels[i]
            if feat.shape[1] != expected_ch:
                raise ValueError(
                    f"Head feature channel mismatch at scale {i}: "
                    f"expected {expected_ch}, got {feat.shape[1]}"
                )

            cls_out = self.cls_convs[i](feat)
            reg_out = self.reg_convs[i](feat)
            
            # Reshape outputs
            bs, _, h, w = cls_out.shape
            cls_out = cls_out.view(bs, 4, self.nc, h, w).permute(0, 1, 3, 4, 2)
            reg_out = reg_out.view(bs, 4, 4, h, w).permute(0, 1, 3, 4, 2)
            
            outputs.append(torch.cat([reg_out, cls_out], dim=-1))
        
        return outputs

class ContextAggregationModule(nn.Module):
    """Aggregates multi-scale context for small object detection"""
    def __init__(self, channels, expansion=4):
        super().__init__()
        mid_channels = channels // expansion
        
        # Multi-scale pooling
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        
        # Context processing - input channels: channels (original) + channels (x1) + channels (x2) + channels (x3) = channels * 4
        self.conv1 = nn.Conv2d(channels * 4, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        
    def forward(self, x):
        bs, c, h, w = x.shape
        
        # Multi-scale pooling
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        
        # Upsample and concatenate
        x1 = F.interpolate(x1, size=(h, w), mode='nearest')
        x2 = F.interpolate(x2, size=(h, w), mode='nearest')
        x3 = F.interpolate(x3, size=(h, w), mode='nearest')
        
        # Feature fusion
        x_cat = torch.cat([x, x1, x2, x3], dim=1)
        
        # Context aggregation
        out = self.act(self.norm(self.conv2(self.act(self.conv1(x_cat)))))
        
        return x + out  # Residual connection

class EnhancedYOLOv8(nn.Module):
    """Complete enhanced YOLOv8 architecture for small object detection"""
    def __init__(self, nc=80, scales='n'):
        super().__init__()
        
        # Backbone configuration (ultra-light for 4GB GPU)
        if scales == 'n':
            backbone_channels = [32, 64, 128, 256, 512]
        elif scales == 's':
            backbone_channels = [48, 96, 192, 384, 768]
        elif scales == 'm':
            backbone_channels = [64, 128, 256, 512, 1024]
        elif scales == 'l':
            backbone_channels = [96, 192, 384, 768, 1536]
        elif scales == 'x':
            backbone_channels = [128, 256, 512, 1024, 2048]
        
        # Enhanced backbone
        self.backbone = EnhancedBackbone(backbone_channels)
        
        # Neck with improved feature pyramid
        self.neck = EnhancedFPN(backbone_channels)
        
        # Specialized head for small objects.
        # EnhancedFPN outputs unified channel width (deepest backbone width) per scale,
        # so head channels should match that width for all three detection scales.
        head_channels = [backbone_channels[-1], backbone_channels[-1], backbone_channels[-1]]
        self.head = SmallObjectDetectionHead(nc, channels=head_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Backbone features
        features = self.backbone(x)
        
        # Neck features
        neck_features = self.neck(features)
        
        # Detection head
        outputs = self.head(neck_features[-3:])  # Last three scales
        
        return outputs

class EnhancedBackbone(nn.Module):
    """Enhanced CSPDarknet backbone with additional small object focus"""
    def __init__(self, channels):
        super().__init__()
        
        # Initial stem with more aggressive downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0]//2, 3, 2, 1),
            nn.BatchNorm2d(channels[0]//2),
            nn.SiLU(),
            EnhancedConv(channels[0]//2, channels[0], 3, 2),
        )
        
        # Enhanced CSP blocks
        self.stage1 = EnhancedCSPBlock(channels[0], channels[1], n=3)
        self.stage2 = EnhancedCSPBlock(channels[1], channels[2], n=6)
        self.stage3 = EnhancedCSPBlock(channels[2], channels[3], n=9)
        self.stage4 = EnhancedCSPBlock(channels[3], channels[4], n=3)
        
        # Additional high-resolution branch for small objects
        self.small_branch = SmallObjectBranch(channels[0])
        
    def forward(self, x):
        x = self.stem(x)
        
        # Main backbone
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        
        # Small object branch (preserves high-resolution features)
        s1 = self.small_branch(x)
        
        return [s1, c1, c2, c3, c4]

class SmallObjectBranch(nn.Module):
    """High-resolution branch specifically for small object features"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = EnhancedConv(channels, channels//2, 1)
        self.conv2 = EnhancedConv(channels//2, channels//2, 3)
        self.conv3 = EnhancedConv(channels//2, channels, 1)
        
        # Dilated convolutions for larger receptive field
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.dilated_conv(x)  # Residual connection
        return x

class EnhancedCSPBlock(nn.Module):
    """Enhanced CSP (Cross Stage Partial) block"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        self.m = nn.Sequential(*[EnhancedConv(c_, c_, 3) for _ in range(n)])
        self.shortcut = shortcut
    
    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class EnhancedFPN(nn.Module):
    """Enhanced Feature Pyramid Network with improved feature fusion"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Build lateral connection layers
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i in range(len(channels)):
            if i < len(channels) - 1:
                lateral_conv = nn.Conv2d(channels[i], channels[-1], kernel_size=1)
                self.lateral_convs.append(lateral_conv)
                
                fpn_conv = nn.Sequential(
                    nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels[-1]),
                    nn.ReLU(inplace=True)
                )
                self.fpn_convs.append(fpn_conv)
    
    def forward(self, features):
        """Forward pass through FPN"""
        if not isinstance(features, (list, tuple)):
            features = [features]
        
        # Build lateral features
        laterals = []
        for feat, lateral_conv in zip(features[:-1], self.lateral_convs):
            laterals.append(lateral_conv(feat))
        
        # Add the deepest feature
        laterals.append(features[-1])
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
        
        # Apply FPN convolutions
        outputs = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            outputs.append(fpn_conv(lateral))
        
        # Add the final deep feature
        outputs.append(laterals[-1])
        
        return outputs


def autopad(k, p=None):
    """Auto-padding calculation"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p