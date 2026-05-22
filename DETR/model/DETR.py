
import torch
import torch.nn as nn
import math
from .FeatureExtractor import FeatureExtractor
from .TransformerEncoder import TransformerEncoder
from .TransformerDecoder import TransformerDecoder
from .DecodeHead import DETRPredictionHeads

class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine positional embeddings used in DETR.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x shape: [Batch, Channels, H, W]
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class DETR(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=100):
        super().__init__()
        self.backbone = FeatureExtractor(hidden_dim)
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.encoder = TransformerEncoder(hidden_dim)
        self.decoder = TransformerDecoder(hidden_dim, num_queries=num_queries)
        self.heads = DETRPredictionHeads(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: [Batch, 3, 800, 800]
        # features shape: [Batch, 256, 25, 25]
        features = self.backbone(x)

        # Generate absolute positional embeddings
        pos_embedding = self.position_embedding(features)
        
        # Transformer Encoder
        memory = self.encoder(features, pos_embedding)
        # memory shape: [625, Batch, 256]
        
        # Transformer Decoder
        decoder_output = self.decoder(memory)
        # decoder_output shape: [Batch, 100, 256]
        
        # Final Prediction
        pred_logits, pred_boxes = self.heads(decoder_output)
        
        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}