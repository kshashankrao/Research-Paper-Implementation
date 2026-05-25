"""
MonoDETR: Monocular 3D Object Detection with Depth Prediction

Simple implementation calling modules step by step.

Pipeline:
    Images → FeatureExtractor → DepthPredictor
                                     ↓
                     VisualEncoder    DepthEncoder
                                     ↓
                         VisualDepthDecoder (fusion)
                                     ↓
                           DecoderHeads (predictions)
"""

import torch
import torch.nn as nn
from typing import Dict

from .FeatureExtractor import FeatureExtractor
from .DepthPredictor import DepthPredictor
from .VisualEncoder import VisualEncoder
from .DepthEncoder import DepthEncoder
from .VisualDepthDecoder import VisualDepthDecoder
from .DecoderHeads import DecoderHeads


class MonoDETR(nn.Module):
    """MonoDETR: 3D object detection from monocular images with depth."""
    
    def __init__(self, num_classes: int = 3, num_queries: int = 100, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(hidden_dim=hidden_dim)
        self.depth_predictor = DepthPredictor(hidden_dim=hidden_dim)
        self.visual_encoder = VisualEncoder(hidden_dim=hidden_dim, num_layers=4)
        self.depth_encoder = DepthEncoder(hidden_dim=hidden_dim, num_layers=4)
        self.visual_depth_decoder = VisualDepthDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=4
        )
        self.decoder_heads = DecoderHeads(hidden_dim=hidden_dim, num_classes=num_classes)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass - call each module step by step.
        
        Args:
            images: [Batch, 3, Height, Width]
        
        Returns:
            Dict with predictions:
            - pred_boxes_2d: [B, 100, 4]
            - pred_boxes_3d: [B, 100, 7]
            - pred_logits: [B, 100, 3]
            - pred_depth: [B, 1, Height, Width]
        """
        # Step 1: Extract visual features
        visual_features = self.feature_extractor(images)
        
        # Step 2: Predict depth map
        depth_map = self.depth_predictor(visual_features)
        
        # Step 3: Encode visual features
        visual_encoded = self.visual_encoder(visual_features, num_queries=100)
        
        # Step 4: Encode depth map
        depth_encoded = self.depth_encoder(depth_map, num_queries=100)
        
        # Step 5: Fuse visual and depth
        fused_features = self.visual_depth_decoder(visual_encoded, depth_encoded)
        
        # Step 6: Generate predictions
        predictions = self.decoder_heads(fused_features)
        
        # Add depth to output
        predictions['pred_depth'] = depth_map
        
        return predictions


if __name__ == '__main__':
    """Test the model."""
    print("Testing MonoDETR...\n")
    
    model = MonoDETR(num_classes=3, num_queries=100)
    print("✓ Model created\n")
    
    images = torch.randn(2, 3, 384, 512)
    print(f"Input: {images.shape}")
    
    with torch.no_grad():
        outputs = model(images)
    
    print(f"\n✓ Forward pass successful!")
    print(f"Outputs:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
