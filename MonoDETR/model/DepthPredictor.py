import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredictor(nn.Module):
    """
    Monocular depth prediction network. It is a depth a decoder.
    
    Takes visual features and progressively upsamples them to predict a dense depth map.
    Uses transposed convolutions followed by regular convolutions to refine predictions.

    It predicts spatial depth map and the loss is computed only for the target objects / foreground. 
    
    Args:
        hidden_dim (int): Dimension of input features. Default: 256
        depth_range (tuple): Min and max depth values (min_depth, max_depth). 
                           Default: (0.1, 80.0)
    """
    
    def __init__(self, hidden_dim: int = 256, depth_range: tuple = (0.1, 80.0)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_depth, self.max_depth = depth_range
        
        # Depth decoder: progressively upsample features
        # Input: [Batch, hidden_dim, H/32, W/32]
        
        # Layer 1: Upsample 16x16 -> 32x32
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Layer 2: Upsample 32x32 -> 64x64
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=hidden_dim // 2,
            out_channels=hidden_dim // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Layer 3: Upsample 64x64 -> 128x128
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=hidden_dim // 4,
            out_channels=hidden_dim // 8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Layer 4: Upsample 128x128 -> 256x256
        self.upconv4 = nn.ConvTranspose2d(
            in_channels=hidden_dim // 8,
            out_channels=hidden_dim // 16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final depth prediction: predict 1 channel (depth value)
        self.depth_head = nn.Conv2d(
            in_channels=hidden_dim // 16,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict depth map from visual features.
        
        Args:
            features (torch.Tensor): Visual features of shape [Batch, hidden_dim, H/32, W/32]
        
        Returns:
            torch.Tensor: Depth map of shape [Batch, 1, Height, Width]
                         Values are constrained to [min_depth, max_depth] using sigmoid scaling
        
        Note:
            The output depth values are scaled to the range [min_depth, max_depth].
            Sigmoid is used to keep values in (0, 1), then scaled accordingly.
        """

        x = self.upconv1(features)
        x = self.conv1(x)
        
        x = self.upconv2(x)
        x = self.conv2(x)
        
        x = self.upconv3(x)
        x = self.conv3(x)
        
        x = self.upconv4(x)
        x = self.conv4(x)
        
        # Predict raw depth values
        depth = self.depth_head(x)
        
        # Scale depth to [min_depth, max_depth] using sigmoid
        # sigmoid maps to (0, 1), then multiply by range and add min_depth
        depth = torch.sigmoid(depth) * (self.max_depth - self.min_depth) + self.min_depth
        
        return depth
