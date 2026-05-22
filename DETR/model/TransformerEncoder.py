import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    """
    The encoder takes the output from the CNN backbone and processes it to produce a set of encoded features.
    It is a stack of 6 layers of multi headed self attention and MLPs.
    Each transformder has a 8 head of self attention.
    
    What is hidden dim ?
    The hidden dimension is the size of the feature vectors that are processed by the transformer.
    """
    
    def __init__(self, hidden_dim=256, nheads=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, pos_embedding):
        # x is the output from CNN: [Batch, 256, H, W]
        B, C, H, W = x.shape
        
        # Transform the 2d image into 1d sequence of pixel features.
        # Flatten spatial dimensions: [Batch, 256, H*W] -> [Batch, H*W, 256]
        x = x.flatten(2).permute(0, 2, 1)
        
        # Add the positional encoding to the pixel features. 
        # This tells the 1D sequence where each pixel originally lived in the 2D image.
        pos_embedding = pos_embedding.flatten(2).permute(0, 2, 1)
        x_with_pos = x + pos_embedding
        
        # PyTorch expects [Seq_Len, Batch, Hidden_Dim]
        # Input shape: [Batch, H*W, 256] -> [H*W, Batch, 256]
        x_with_pos = x_with_pos.permute(1, 0, 2)
        
        # The input and output shape of the enccoder are the same
        # Output shape: [H*W, Batch, 256]
        encoder_features = self.encoder(x_with_pos)
        
        return encoder_features