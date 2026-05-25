"""
Tests for DepthEncoder module.
"""

import pytest
import torch
from model.DepthEncoder import DepthEncoder


class TestDepthEncoder:
    """Test suite for DepthEncoder."""
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        encoder = DepthEncoder()
        assert encoder.hidden_dim == 256
        assert len(encoder.layers) == 4
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        encoder = DepthEncoder(hidden_dim=512, num_layers=6, num_heads=16)
        assert encoder.hidden_dim == 512
        assert len(encoder.layers) == 6
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = DepthEncoder(hidden_dim=256, num_layers=4)
        
        # Input: [Batch, 1, Height, Width] - depth map
        depth_map = torch.randn(2, 1, 384, 512)
        
        # Forward pass
        output = encoder(depth_map, num_queries=100)
        
        # Output should be [Batch, num_queries, hidden_dim]
        assert output.shape == (2, 100, 256)
    
    def test_depth_projection(self):
        """Test that depth projection layer exists."""
        encoder = DepthEncoder(hidden_dim=256)
        assert hasattr(encoder, 'depth_projection')
        
        # Depth projection should convert 1-channel to hidden_dim channels
        depth_map = torch.randn(2, 1, 256, 256)
        projected = encoder.depth_projection(depth_map)
        
        assert projected.shape == (2, 256, 256, 256)
    
    def test_different_spatial_sizes(self):
        """Test with different depth map sizes."""
        encoder = DepthEncoder(hidden_dim=256, num_layers=4)
        
        test_cases = [
            (2, 1, 128, 128),
            (2, 1, 384, 512),
            (2, 1, 480, 640),
        ]
        
        for batch, channels, h, w in test_cases:
            depth_map = torch.randn(batch, channels, h, w)
            output = encoder(depth_map, num_queries=100)
            
            assert output.shape == (batch, 100, 256)
    
    def test_different_num_queries(self):
        """Test extracting different number of queries."""
        encoder = DepthEncoder(hidden_dim=256, num_layers=4)
        depth_map = torch.randn(2, 1, 384, 512)
        
        for num_queries in [50, 100, 200]:
            output = encoder(depth_map, num_queries=num_queries)
            assert output.shape == (2, num_queries, 256)
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow through encoder."""
        depth_map = torch.randn(2, 1, 384, 512, requires_grad=True)
        encoder = DepthEncoder(hidden_dim=256, num_layers=4)
        
        output = encoder(depth_map, num_queries=100)
        loss = output.sum()
        loss.backward()
        
        assert depth_map.grad is not None
        assert depth_map.grad.shape == depth_map.shape
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        encoder = DepthEncoder(hidden_dim=256, num_layers=4)
        
        for batch_size in [1, 2, 4, 8]:
            depth_map = torch.randn(batch_size, 1, 384, 512)
            output = encoder(depth_map, num_queries=100)
            
            assert output.shape == (batch_size, 100, 256)
    
    def test_eval_mode(self):
        """Test encoder in evaluation mode."""
        encoder = DepthEncoder(hidden_dim=256, num_layers=4, dropout=0.5)
        encoder.eval()
        
        depth_map = torch.randn(2, 1, 384, 512)
        output = encoder(depth_map, num_queries=100)
        
        assert output.shape == (2, 100, 256)
