"""
CrossAttention: Cross-attention mechanism for feature fusion.

This module implements cross-attention to fuse features from two different sources,
commonly used for combining visual and depth information.
"""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism to fuse two feature sequences.
    
    Allows one sequence (query) to attend to another sequence (key/value),
    useful for fusing visual and depth features.
    
    Args:
        hidden_dim (int): Feature dimension
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query projection (from first feature source)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        # Key and value projections (from second feature source)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query_features: torch.Tensor, key_value_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention from query features to key/value features.
        
        Args:
            query_features (torch.Tensor): Query features of shape [Batch, Seq_len, hidden_dim]
            key_value_features (torch.Tensor): Key/value features of shape [Batch, Seq_len, hidden_dim]
        
        Returns:
            torch.Tensor: Attended features of shape [Batch, Seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = query_features.shape
        
        # Project to Q, K, V
        Q = self.query_proj(query_features)
        K = self.key_proj(key_value_features)
        V = self.value_proj(key_value_features)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output
