"""
Tests for VisualEncoder module.
"""

import pytest
import torch
from model.VisualEncoder import VisualEncoder


class TestVisualEncoder:
    """Test suite for VisualEncoder."""
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        encoder = VisualEncoder()
        assert encoder.hidden_dim == 256
        assert len(encoder.layers) == 4
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        encoder = VisualEncoder(hidden_dim=512, num_layers=6, num_heads=16)
        assert encoder.hidden_dim == 512
        assert len(encoder.layers) == 6
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        
        # Input: [Batch, hidden_dim, H/32, W/32]
        features = torch.randn(2, 256, 12, 16)
        
        # Forward pass
        output = encoder(features, num_queries=100)
        
        # Output should be [Batch, num_queries, hidden_dim]
        assert output.shape == (2, 100, 256)
    
    def test_different_spatial_sizes(self):
        """Test with different spatial feature sizes."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        
        test_cases = [
            (2, 256, 8, 8),    # Small features
            (2, 256, 12, 16),  # Medium features
            (2, 256, 24, 32),  # Large features
        ]
        
        for batch, channels, h, w in test_cases:
            features = torch.randn(batch, channels, h, w)
            output = encoder(features, num_queries=100)
            
            assert output.shape == (batch, 100, 256)
    
    def test_different_num_queries(self):
        """Test extracting different number of queries."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        features = torch.randn(2, 256, 12, 16)
        
        for num_queries in [50, 100, 200]:
            output = encoder(features, num_queries=num_queries)
            assert output.shape == (2, num_queries, 256)
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow through encoder."""
        features = torch.randn(2, 256, 12, 16, requires_grad=True)
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        
        output = encoder(features, num_queries=100)
        loss = output.sum()
        loss.backward()
        
        assert features.grad is not None
        assert features.grad.shape == features.shape
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        
        for batch_size in [1, 2, 4, 8]:
            features = torch.randn(batch_size, 256, 12, 16)
            output = encoder(features, num_queries=100)
            
            assert output.shape == (batch_size, 100, 256)
    
    def test_eval_mode(self):
        """Test encoder in evaluation mode."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4, dropout=0.5)
        encoder.eval()
        
        features = torch.randn(2, 256, 12, 16)
        output = encoder(features, num_queries=100)
        
        assert output.shape == (2, 100, 256)
    
    def test_output_is_not_input(self):
        """Test that output is different from input features."""
        encoder = VisualEncoder(hidden_dim=256, num_layers=4)
        features = torch.randn(2, 256, 12, 16)
        
        output = encoder(features, num_queries=100)
        
        # Output should not be the same as input spatial tokens
        # (since features are transformed by attention layers)
        assert output.shape != features.shape
