"""
ALGA-DMA Cloud Classification Model

This module contains the implementation of the ALGA (Adaptive Local-Global Attention) 
and DMA (Dynamic Multi-Head Attention) model for cloud classification.

Author: Your Name
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
import math


class MixedConv2d(nn.Module):
    """Mixed Convolution layer with multiple kernel sizes"""
    
    def __init__(self, in_channels, out_channels=256, kernel_sizes=[1, 3, 5], stride=1, padding=1):
        super(MixedConv2d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(kernel_sizes), kernel_size=k, 
                     stride=stride, padding=k//2) for k in kernel_sizes
        ])
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DynamicMultiHeadAttention(nn.Module):
    """Dynamic Multi-Head Attention mechanism"""
    
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(DynamicMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_into_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        # Split into heads
        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights


class ALGANet(nn.Module):
    """Adaptive Local-Global Attention Network"""
    
    def __init__(self, channels, enhance_local_features=True):
        super(ALGANet, self).__init__()
        self.channels = channels
        self.enhance_local_features = enhance_local_features
        
        # Local attention components
        self.local_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.local_bn = nn.BatchNorm2d(channels)
        
        # Global attention components
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Local attention
        local_features = self.local_attention(x)
        
        # Global attention
        global_features = self.global_attention(x)
        
        # Combine local and global features
        if self.enhance_local_features:
            output = local_features * global_features + x
        else:
            output = local_features + global_features
            
        return output
    
    def local_attention(self, x):
        local_out = self.local_conv(x)
        local_out = self.local_bn(local_out)
        return torch.sigmoid(local_out)
    
    def global_attention(self, x):
        b, c, _, _ = x.size()
        global_out = self.global_pool(x).view(b, c)
        global_out = self.global_fc(global_out).view(b, c, 1, 1)
        return global_out.expand_as(x)


class ALGADMAModel(nn.Module):
    """Complete ALGA-DMA model for cloud classification"""
    
    def __init__(self, num_classes=7):
        super(ALGADMAModel, self).__init__()
        
        # Backbone: DenseNet121
        self.backbone = densenet121(pretrained=True)
        backbone_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Mixed convolution layer
        self.mixed_conv = MixedConv2d(backbone_features, 512)
        
        # Depthwise separable convolution
        self.depthwise_conv = DepthwiseSeparableConv2d(512, 512, kernel_size=3)
        
        # ALGA module
        self.alga = ALGANet(512)
        
        # DMA module
        self.dma = DynamicMultiHeadAttention(num_heads=8, d_model=512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Reshape for convolution operations
        b, c = features.size()
        h = w = int((c / 1024) ** 0.5) if c > 1024 else 1
        features = features.view(b, 1024, h, w)
        
        # Apply mixed convolution
        features = self.mixed_conv(features)
        
        # Apply depthwise separable convolution
        features = self.depthwise_conv(features)
        
        # Apply ALGA
        features = self.alga(features)
        
        # Global pooling and flatten
        pooled_features = self.global_pool(features).flatten(1)
        
        # Prepare for DMA (reshape to sequence format)
        seq_features = pooled_features.unsqueeze(1)  # (batch, 1, features)
        
        # Apply DMA
        dma_output, _ = self.dma(seq_features, seq_features, seq_features)
        dma_output = dma_output.squeeze(1)  # Remove sequence dimension
        
        # Classification
        output = self.classifier(dma_output)
        
        return output


def create_model(num_classes=7, pretrained=True):
    """
    Create ALGA-DMA model
    
    Args:
        num_classes (int): Number of cloud classes
        pretrained (bool): Use pretrained backbone
        
    Returns:
        ALGADMAModel: The complete model
    """
    return ALGADMAModel(num_classes=num_classes)


if __name__ == "__main__":
    # Test model creation
    model = create_model(num_classes=7)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")