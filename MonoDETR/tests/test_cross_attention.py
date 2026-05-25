"""
Tests for CrossAttention module.
"""

import pytest
import torch
from model.CrossAttention import CrossAttention


class TestCrossAttention:
    """Test suite for CrossAttention."""
    
    def test_initialization(self):
        """Test initialization with valid parameters."""
        cross_attn = CrossAttention(hidden_dim=256, num_heads=8)
        assert cross_attn.hidden_dim == 256
        assert cross_attn.num_heads == 8
        assert cross_attn.head_dim == 32
    
    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises assertion error."""
        with pytest.raises(AssertionError):
            CrossAttention(hidden_dim=256, num_heads=7)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, seq_len, hidden_dim = 2, 100, 256
        cross_attn = CrossAttention(hidden_dim=hidden_dim, num_heads=8)
        
        query = torch.randn(batch_size, seq_len, hidden_dim)
        key_value = torch.randn(batch_size, seq_len, hidden_dim)
        
        output = cross_attn(query, key_value)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
    
    def test_different_sequence_lengths(self):
        """Test cross-attention with different query and key/value lengths."""
        cross_attn = CrossAttention(hidden_dim=256, num_heads=8)
        
        batch_size, hidden_dim = 2, 256
        query = torch.randn(batch_size, 100, hidden_dim)
        key_value = torch.randn(batch_size, 192, hidden_dim)
        
        # Should work even with different sequence lengths in key/value
        output = cross_attn(query, key_value)
        assert output.shape == (batch_size, 100, hidden_dim)
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow through cross-attention."""
        query = torch.randn(2, 100, 256, requires_grad=True)
        key_value = torch.randn(2, 100, 256, requires_grad=True)
        
        cross_attn = CrossAttention(hidden_dim=256, num_heads=8)
        output = cross_attn(query, key_value)
        
        loss = output.sum()
        loss.backward()
        
        assert query.grad is not None
        assert key_value.grad is not None
    
    def test_visual_to_depth_fusion(self):
        """Test cross-attention for visual-to-depth fusion."""
        batch_size, num_queries, hidden_dim = 2, 100, 256
        
        visual_features = torch.randn(batch_size, num_queries, hidden_dim)
        depth_features = torch.randn(batch_size, num_queries, hidden_dim)
        
        cross_attn = CrossAttention(hidden_dim=hidden_dim, num_heads=8)
        
        # Visual attends to depth
        fused = cross_attn(visual_features, depth_features)
        
        assert fused.shape == visual_features.shape
        assert not torch.allclose(fused, visual_features)  # Should be modified
    
    def test_batch_processing(self):
        """Test processing multiple batches."""
        cross_attn = CrossAttention(hidden_dim=128, num_heads=4)
        
        for batch_size in [1, 2, 4, 8]:
            query = torch.randn(batch_size, 64, 128)
            key_value = torch.randn(batch_size, 64, 128)
            
            output = cross_attn(query, key_value)
            assert output.shape == (batch_size, 64, 128)
    
    def test_different_head_configs(self):
        """Test different head configurations."""
        configs = [(256, 8), (512, 16), (128, 4)]
        
        for hidden_dim, num_heads in configs:
            cross_attn = CrossAttention(hidden_dim=hidden_dim, num_heads=num_heads)
            
            query = torch.randn(2, 100, hidden_dim)
            key_value = torch.randn(2, 100, hidden_dim)
            
            output = cross_attn(query, key_value)
            assert output.shape == (2, 100, hidden_dim)
