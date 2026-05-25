"""
VisualEncoder: Encode visual features using transformer self-attention.

This module takes dense CNN features and encodes them using multi-head self-attention,
capturing spatial relationships and global context in the image.

Input: [Batch, hidden_dim, Height/32, Width/32]
Output: [Batch, num_queries, hidden_dim]
"""

import torch
import torch.nn as nn
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class VisualEncoder(nn.Module):
    """
    Visual feature encoder using transformer self-attention layers.
    
    Takes dense CNN features (spatially organized), flattens them to a sequence,
    and applies multi-layer self-attention to capture global context.
    
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
        
        # Stack of encoder layers
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
    
    def forward(self, features: torch.Tensor, num_queries: int = 100) -> torch.Tensor:
        """
        Encode visual features using self-attention.
        
        Args:
            features (torch.Tensor): Visual features of shape [Batch, hidden_dim, H/32, W/32]
            num_queries (int): Number of feature tokens to extract. Default: 100
        
        Returns:
            torch.Tensor: Encoded features of shape [Batch, num_queries, hidden_dim]
        
        Note:
            Returns top num_queries tokens from encoded spatial features.
            These are used as key/value in decoder cross-attention with learnable object queries.
        """
        batch_size, channels, height, width = features.shape
        
        # Flatten spatial dimensions: [Batch, hidden_dim, H/32, W/32] -> [Batch, H/32*W/32, hidden_dim]
        x = features.flatten(2).transpose(1, 2)  # [Batch, spatial_len, hidden_dim]
        
        # Apply self-attention layers
        for layer in self.layers:
            # Self-attention with residual connection and layer norm
            attn_out = layer['self_attn'](x)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward with residual connection and layer norm
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
        
        # Extract top num_queries tokens (for efficiency and consistency)
        # Shape: [Batch, num_queries, hidden_dim]
        visual_features = x[:, :num_queries, :]
        
        return visual_features
