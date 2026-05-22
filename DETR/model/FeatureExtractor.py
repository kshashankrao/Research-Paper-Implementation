import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """
    Use CNN to extract the features from the input image.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Remove the last 2 layers: Average Pooling and Fully Connected layer
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    
        # Project the output to the output channel dimension
        self.conv = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x):
        # Input dim: [Batch, 3, Height, Width]
        features = self.backbone(x)
        
        # Output dim: [Batch, 256, H/32, W/32]
        projected_features = self.conv(features)
        
        return projected_features