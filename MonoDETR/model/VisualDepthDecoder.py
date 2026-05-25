"""
VisualDepthDecoder: Fuse visual and depth features for 3D object detection.

This module combines encoded visual features with encoded depth features
using learnable object queries and cross-attention/self-attention mechanisms,
creating a fused representation that leverages both appearance and geometric information.

Input: Visual features [Batch, num_queries, hidden_dim] (from VisualEncoder)
       Depth features [Batch, num_queries, hidden_dim] (from DepthEncoder)
Output: [Batch, num_queries, hidden_dim] (learnable object queries fused with visual/depth)  
"""

import torch
import torch.nn as nn
from .CrossAttention import CrossAttention
from .MultiHeadSelfAttention import MultiHeadSelfAttention


class VisualDepthDecoder(nn.Module):
    """
    Decoder that fuses visual and depth features.
    
    Uses learnable object queries and alternating cross-attention and self-attention 
    layers to progressively fuse visual and depth information for improved 3D object detection.
    
    Args:
        hidden_dim (int): Feature dimension. Default: 256
        num_queries (int): Number of learnable object queries. Default: 100
        num_layers (int): Number of fusion layers. Default: 4
        num_heads (int): Number of attention heads. Default: 8
        mlp_dim (int): Dimension of feed-forward network. Default: 2048
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, hidden_dim: int = 256, num_queries: int = 100, num_layers: int = 4,
                 num_heads: int = 8, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim # Size of each token embedding
        self.num_queries = num_queries # Number of learnable object queries (tokens)
        
        # Learnable object queries. 
        # It captures the object representations from visual and depth features
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Fusion layers with cross-attention and self-attention
        # Stack the indivdual layers of the decoderof the transformer
        # Each layer consists of:
        # 1. Cross-attention from object queries to visual features
        # 2. Cross-attention from object queries to depth features (symmetric fusion)
        # 3. Self-attention on fused representation
        # 4. Feed-forward network for further processing
        self.fusion_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Visual attends to depth
                'cross_attn_vd': CrossAttention(hidden_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                # Depth attends to visual (symmetric fusion)
                'cross_attn_dv': CrossAttention(hidden_dim, num_heads, dropout),
                'norm2': nn.LayerNorm(hidden_dim),
                # Self-attention on fused features
                'self_attn': MultiHeadSelfAttention(hidden_dim, num_heads, dropout),
                'norm3': nn.LayerNorm(hidden_dim),
                # Feed-forward network
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, mlp_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm4': nn.LayerNorm(hidden_dim)
            })
            self.fusion_layers.append(layer)
    
    def forward(self, visual_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and depth features with learnable object queries.
        
        Args:
            visual_features (torch.Tensor): Encoded visual features from VisualEncoder of shape [Batch, num_queries, hidden_dim]
            depth_features (torch.Tensor): Encoded depth features from DepthEncoder of shape [Batch, num_queries, hidden_dim]
        
        Returns:
            torch.Tensor: Fused features of shape [Batch, num_queries, hidden_dim]
                         where num_queries are learnable embeddings
        
        Note:
            The fusion process for each layer:
            1. Object queries cross-attend to depth features
            2. Self-attention on the depth-fused representation
            3. Cross-attend to visual features
            4. Feed-forward network for aggregation
        """
        batch_size = visual_features.shape[0]
        
        # Initialize with learnable object queries
        # Shape: [num_queries, hidden_dim]
        query_indices = torch.arange(self.num_queries, device=visual_features.device)
        object_queries = self.query_embed(query_indices)
        
        # Expand to batch: [Batch, num_queries, hidden_dim]
        object_queries = object_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        for layer in self.fusion_layers:
            # Fuse object queries with DEPTH information
            # Object queries (as Q) attend to depth features (as K, V)
            # Output: [Batch, num_queries, hidden_dim]
            depth_context = layer['cross_attn_vd'](object_queries, depth_features)
            fused_representation = layer['norm1'](fused_representation + depth_context)
            
            # Refine with SELF-ATTENTION
            # Let queries interact with each other after incorporating depth
            # Output: [Batch, num_queries, hidden_dim]
            self_refined = layer['self_attn'](fused_representation)
            fused_representation = layer['norm2'](fused_representation + self_refined)
            
            # Fuse with VISUAL information
            # Depth-refined queries (as Q) attend to visual features (as K, V)
            # Output: [Batch, num_queries, hidden_dim]
            visual_context = layer['cross_attn_dv'](fused_representation, visual_features)
            fused_representation = layer['norm3'](fused_representation + visual_context)
            
            # Aggregate with FEED-FORWARD
            # MLP processes the fused representation
            # Output: [Batch, num_queries, hidden_dim]
            aggregated = layer['mlp'](fused_representation)
            fused_representation = layer['norm4'](fused_representation + aggregated)
        
        return fused_representation
