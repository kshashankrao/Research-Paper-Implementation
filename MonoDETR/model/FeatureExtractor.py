"""
FeatureExtractor: Extract visual features from RGB images using ResNet50 backbone.

This module implements a CNN-based feature extraction network that takes RGB images
as input and produces dense feature maps. The extracted features are passed to both
the visual encoder and the depth predictor.

Input: [Batch, 3, Height, Width]
Output: [Batch, hidden_dim, Height/32, Width/32]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """
    CNN-based feature extractor using ResNet50 backbone.
    
    The backbone extracts multi-scale features from the input image.
    A 1x1 convolution projects the features to the desired hidden dimension.
    
    Args:
        hidden_dim (int): Dimension of output features. Default: 256
        backbone_name (str): Name of the backbone network. Default: 'resnet50'
        pretrained (bool): Whether to use pretrained weights. Default: True
    """
    
    def __init__(self, hidden_dim: int = 256, backbone_name: str = 'resnet50', 
                 pretrained: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Load pretrained ResNet50 and remove classification head
        # Remove last 2 layers: AdaptiveAvgPool2d and Linear
        # ResNet50 outputs 2048 channels at layer4
        if backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
        
        # Project backbone output to hidden_dim
        self.projection = nn.Conv2d(
            in_channels=backbone_out_channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        
        self.bn = nn.BatchNorm2d(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        """
        Extract features from input images.
        
        Args:
            x (torch.Tensor): Input images of shape [Batch, 3, Height, Width]
                             Pixel values should be normalized to [-1, 1] or [0, 1]
        
        Returns:
            torch.Tensor: Extracted features of shape [Batch, hidden_dim, H/32, W/32]
                         The spatial resolution is reduced by 32x due to ResNet50's 
                         strided convolutions (stride 2 at each layer)
        """
        # Input shape: [Batch, 3, Height, Width]
        # Shape after backbone: [Batch, 2048, H/32, W/32]
        features = self.backbone(x)
        
        # Output Shape: [Batch, hidden_dim, H/32, W/32]
        features = self.projection(features)
                
        return features
