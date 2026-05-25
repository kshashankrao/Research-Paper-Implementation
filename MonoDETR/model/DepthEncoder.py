"""
DepthEncoder: Encode depth information using transformer self-attention.

This module takes depth maps and encodes them using multi-head self-attention,
similar to VisualEncoder but operating on depth features.

Input: [Batch, 1, Height, Width]
Output: [Batch, num_queries, hidden_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class DepthEncoder(nn.Module):
    """
    Depth feature encoder using transformer self-attention layers.
    
    Takes depth maps, projects them to hidden_dim, and applies multi-layer
    self-attention to capture depth-based spatial relationships.
    
    Args:
        hidden_dim (int): Feature dimension. Default: 256
        num_layers (int): Number of encoder layers. Default: 4
        num_heads (int): Number of attention heads. Default: 8
        mlp_dim (int): Dimension of feed-forward network. Default: 2048
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, hidden_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 8, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input is [Batch, 1, H, W], we'll project to [Batch, hidden_dim, H, W]
        self.depth_projection = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Stack of encoder layers of transformer 
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'self_attn': MultiHeadSelfAttention(hidden_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, mlp_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            })
            self.layers.append(layer)
    
    def forward(self, depth: torch.Tensor, num_queries: int = 100) -> torch.Tensor:
        """
        Encode depth features using self-attention.
        
        Args:
            depth (torch.Tensor): Depth maps of shape [Batch, 1, Height, Width]
            num_queries (int): Number of feature tokens to extract. Default: 100
        
        Returns:
            torch.Tensor: Encoded depth features of shape [Batch, num_queries, hidden_dim]
        
        Note:
            Returns top num_queries tokens from encoded depth features.
            These are used as key/value in decoder cross-attention with learnable object queries.
        """
        batch_size, _, height, width = depth.shape
        
        # Project depth maps to hidden_dim
        # Input shape: [Batch, 1, Height, Width]
        # Output shape: [Batch, hidden_dim, height, width]
        x = self.depth_projection(depth)
        
        # Flatten spatial dimensions to sequence
        # Output Shape: [Batch, height*width, hidden_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # Apply self-attention layers
        for layer in self.layers:
            # Self-attention with residual connection and layer norm
            attn_out = layer['self_attn'](x)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward with residual connection and layer norm
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
        
        # Extract top num_queries tokens (for efficiency and consistency)
        # Output Shape: [Batch, num_queries, hidden_dim]
        depth_features = x[:, :num_queries, :]
        
        return depth_features
