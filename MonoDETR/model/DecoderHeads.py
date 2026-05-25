"""
DecoderHeads: Prediction heads for 3D object detection and classification.

This module contains the final prediction heads that take the fused visual-depth
features and produce outputs for:
- 2D bounding box predictions (for the projected 2D box)
- 3D bounding box predictions (center, dimensions, orientation)
- Object class predictions

Input: [Batch, num_queries, hidden_dim]
Output: 2D boxes [Batch, num_queries, 4]
        3D boxes [Batch, num_queries, 7] (3D center + dimensions + rotation)
        Class logits [Batch, num_queries, num_classes]
"""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Multi-layer perceptron block with ReLU activation and layer norm.
    
    Args:
        in_dim (int): Input dimension
        hidden_dim (int): Hidden dimension
        out_dim (int): Output dimension
        num_layers (int): Number of layers. Default: 3
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
                break
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP block."""
        return self.net(x)


class BBox2DHead(nn.Module):
    """
    2D bounding box prediction head.
    
    Predicts 2D bounding boxes in the format [x_min, y_min, x_max, y_max]
    (normalized coordinates in [0, 1]).
    
    Args:
        hidden_dim (int): Input feature dimension. Default: 256
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.mlp = MLPBlock(hidden_dim, hidden_dim, 4, num_layers=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict 2D bounding boxes.
        
        Args:
            x (torch.Tensor): Features of shape [Batch, num_queries, hidden_dim]
        
        Returns:
            torch.Tensor: 2D boxes of shape [Batch, num_queries, 4]
                         Values are constrained to [0, 1] using sigmoid
        """
        # Predict raw 2D box coordinates
        boxes_2d = self.mlp(x)
        
        # Constrain to [0, 1] using sigmoid
        boxes_2d = torch.sigmoid(boxes_2d)
        
        return boxes_2d


class BBox3DHead(nn.Module):
    """
    3D bounding box prediction head.
    
    Predicts 3D bounding boxes with:
    - 3D center location in camera coordinates: (x, y, z) - 3 values
    - Dimensions (length, width, height): (l, w, h) - 3 values
    - Rotation angle around vertical axis: (alpha or ry) - 1 value
    
    Total: 7 values per box
    
    Args:
        hidden_dim (int): Input feature dimension. Default: 256
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 3D center location MLP
        self.center_mlp = MLPBlock(hidden_dim, hidden_dim, 3, num_layers=3)
        
        # Dimensions MLP (must be positive)
        self.dims_mlp = MLPBlock(hidden_dim, hidden_dim, 3, num_layers=3)
        
        # Rotation angle MLP (angle in [-pi, pi])
        self.rotation_mlp = MLPBlock(hidden_dim, hidden_dim, 1, num_layers=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict 3D bounding boxes.
        
        Args:
            x (torch.Tensor): Features of shape [Batch, num_queries, hidden_dim]
        
        Returns:
            torch.Tensor: 3D boxes of shape [Batch, num_queries, 7]
                         Format: [cx, cy, cz, l, w, h, rotation]
                         - Center coordinates are in camera coordinate system
                         - Dimensions are constrained to be positive using softplus
                         - Rotation is constrained to [-pi, pi] using tanh
        """
        # Predict 3D center location
        # Shape: [Batch, num_queries, 3]
        center_3d = self.center_mlp(x)
        
        # Predict dimensions (length, width, height)
        # Use softplus to ensure positive values: softplus(x) = log(1 + exp(x))
        # Shape: [Batch, num_queries, 3]
        dims_raw = self.dims_mlp(x)
        dims = torch.nn.functional.softplus(dims_raw)
        
        # Predict rotation angle
        # Use tanh to constrain to [-1, 1], then scale to [-pi, pi]
        # Shape: [Batch, num_queries, 1]
        rotation_raw = self.rotation_mlp(x)
        rotation = torch.atan(torch.tanh(rotation_raw) * 1000)  # Approximate pi
        
        # Concatenate all predictions: [center (3) + dims (3) + rotation (1)] = 7
        # Shape: [Batch, num_queries, 7]
        boxes_3d = torch.cat([center_3d, dims, rotation], dim=-1)
        
        return boxes_3d


class ClassificationHead(nn.Module):
    """
    Object classification head.
    
    Predicts class logits for each object query.
    
    Args:
        hidden_dim (int): Input feature dimension. Default: 256
        num_classes (int): Number of object classes. Default: 3 (car, pedestrian, cyclist)
    """
    
    def __init__(self, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        # Add 1 for "no object" class (background)
        self.mlp = MLPBlock(hidden_dim, hidden_dim, num_classes + 1, num_layers=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits.
        
        Args:
            x (torch.Tensor): Features of shape [Batch, num_queries, hidden_dim]
        
        Returns:
            torch.Tensor: Class logits of shape [Batch, num_queries, num_classes + 1]
                         Including background class at index 0
        """
        logits = self.mlp(x)
        return logits


class DecoderHeads(nn.Module):
    """
    Combined prediction heads for MonoDETR.
    
    Produces simultaneous predictions for:
    - 2D bounding boxes (for monocular detection)
    - 3D bounding boxes (with depth information)
    - Object classification
    
    Args:
        hidden_dim (int): Feature dimension. Default: 256
        num_classes (int): Number of object classes. Default: 3
    """
    
    def __init__(self, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.bbox_2d_head = BBox2DHead(hidden_dim)
        self.bbox_3d_head = BBox3DHead(hidden_dim)
        self.class_head = ClassificationHead(hidden_dim, num_classes)
    
    def forward(self, fused_features: torch.Tensor) -> dict:
        """
        Predict all outputs from fused visual-depth features.
        
        Args:
            fused_features (torch.Tensor): Fused features of shape [Batch, num_queries, hidden_dim]
        
        Returns:
            dict: Dictionary containing:
                - 'pred_boxes_2d': 2D bounding boxes [Batch, num_queries, 4]
                - 'pred_boxes_3d': 3D bounding boxes [Batch, num_queries, 7]
                - 'pred_logits': Class logits [Batch, num_queries, num_classes + 1]
        
        Example:
            >>> outputs = decoder_heads(fused_features)
            >>> print(outputs['pred_boxes_2d'].shape)  # [B, N, 4]
            >>> print(outputs['pred_boxes_3d'].shape)  # [B, N, 7]
            >>> print(outputs['pred_logits'].shape)    # [B, N, 4]
        """
        predictions = {
            'pred_boxes_2d': self.bbox_2d_head(fused_features),
            'pred_boxes_3d': self.bbox_3d_head(fused_features),
            'pred_logits': self.class_head(fused_features)
        }
        
        return predictions
