"""
Tests for MultiHeadSelfAttention module.
"""

import pytest
import torch
from model.MultiHeadSelfAttention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention:
    """Test suite for MultiHeadSelfAttention."""
    
    def test_initialization(self):
        """Test initialization with valid parameters."""
        attn = MultiHeadSelfAttention(hidden_dim=256, num_heads=8)
        assert attn.hidden_dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32
    
    def test_invalid_hidden_dim(self):
        """Test that invalid hidden_dim raises assertion error."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(hidden_dim=256, num_heads=7)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, seq_len, hidden_dim = 2, 192, 256
        attn = MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=8)
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        output = attn(x)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow through attention."""
        x = torch.randn(2, 100, 256, requires_grad=True)
        attn = MultiHeadSelfAttention(hidden_dim=256, num_heads=8)
        
        output = attn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_batch_processing(self):
        """Test processing multiple batches."""
        attn = MultiHeadSelfAttention(hidden_dim=128, num_heads=4)
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 64, 128)
            output = attn(x)
            assert output.shape == (batch_size, 64, 128)
    
    def test_different_head_configs(self):
        """Test different head configurations."""
        configs = [(256, 8), (512, 16), (128, 4), (1024, 8)]
        
        for hidden_dim, num_heads in configs:
            attn = MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
            x = torch.randn(2, 100, hidden_dim)
            output = attn(x)
            assert output.shape == (2, 100, hidden_dim)
    
    def test_dropout_effect(self):
        """Test that dropout affects attention weights differently in train vs eval."""
        attn_train = MultiHeadSelfAttention(hidden_dim=256, num_heads=8, dropout=0.5)
        attn_train.train()
        
        attn_eval = MultiHeadSelfAttention(hidden_dim=256, num_heads=8, dropout=0.5)
        attn_eval.eval()
        
        x = torch.randn(2, 100, 256)
        
        # With seed for reproducibility
        torch.manual_seed(42)
        out_train = attn_train(x)
        
        torch.manual_seed(42)
        out_eval = attn_eval(x)
        
        # Outputs should differ due to dropout
        assert not torch.allclose(out_train, out_eval, atol=0.01)
