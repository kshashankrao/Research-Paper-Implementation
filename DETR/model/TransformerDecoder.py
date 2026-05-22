import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_layers=6, num_queries=100):
        super().__init__()
        
        # The object queries are learnable embeddings. These weights are updated during backpropagation.
        # The size of the object queries is a hyperparameter. 
        # It is usually set to Maximum number of predictions by the modelx Hidden dim = (100 x 256)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, dim_feedforward=2048, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Input shape: [H*W, Batch, 256]
        B = x.shape[1]  
        
        # Retrieve the learnable query embeddings. Object queries shape: [100, 256]
        query_pos = self.query_embed.weight
        
        # Expand them to match the batch size: [100, Batch, 256]
        query_pos = query_pos.unsqueeze(1).repeat(1, B, 1)
        
        # Initialize the output of the decoder with zeros.
        # The decoder will learn to attend to the relevant parts of the encoder output and produce meaningful predictions. 
        target = torch.zeros_like(query_pos)
        
        # Cross attention is between the object queries and the encoder output. This will select the relevant features from the encoder output for each object query.
        # The object queries will be updated during backpropation such that it will learn to attend to the relevant parts of the encoder output and produce meaningful predictions.
        out = self.decoder(tgt=target, memory=x, tgt_key_padding_mask=None, query_pos=query_pos)
        
        # Output shape: [100, Batch, 256] --> [Batch, 100, 256]
        return out.permute(1, 0, 2)