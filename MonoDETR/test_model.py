"""
Simple end-to-end test of MonoDETR model.

Shows how to:
1. Create the model
2. Create dummy input
3. Run forward pass through all components
4. View outputs at each step
"""

import torch
from model.MonoDETR import MonoDETR


def test_end_to_end():
    """Test the model end-to-end without training."""
    
    print("=" * 70)
    print("MonoDETR End-to-End Test")
    print("=" * 70)
    
    # Create model
    print("\n1. Creating model...")
    model = MonoDETR(num_classes=3, num_queries=100, hidden_dim=256)
    model.eval()  # Set to evaluation mode
    print("   ✓ Model created\n")
    
    # Create dummy input
    print("2. Creating dummy input...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 384, 512)
    print(f"   Input shape: {images.shape}")
    print(f"   (Batch: {batch_size}, Channels: 3, Height: 384, Width: 512)\n")
    
    # Forward pass
    print("3. Running forward pass through model...\n")
    
    with torch.no_grad():  # No gradient computation for testing
        outputs = model(images)
    
    # Display outputs
    print("4. Model outputs:\n")
    
    for key, value in outputs.items():
        print(f"   {key}:")
        print(f"      Shape: {value.shape}")
        print(f"      dtype: {value.dtype}")
        print(f"      device: {value.device}\n")
    
    # Explain each output
    print("5. Output Descriptions:\n")
    print("   pred_boxes_2d: 2D bounding boxes (top-left and bottom-right corners)")
    print("      Shape [2, 100, 4] means: [batch_size, num_queries, (x_min, y_min, x_max, y_max)]\n")
    
    print("   pred_boxes_3d: 3D bounding boxes in camera coordinates")
    print("      Shape [2, 100, 7] means: [batch_size, num_queries, (cx, cy, cz, l, w, h, rotation)]\n")
    
    print("   pred_logits: Classification logits for each object query")
    print("      Shape [2, 100, 4] means: [batch_size, num_queries, num_classes]\n")
    
    print("   pred_depth: Predicted depth map for the scene")
    print("      Shape [2, 1, 384, 512] means: [batch_size, 1_channel, height, width]\n")
    
    print("=" * 70)
    print("✓ End-to-end test completed successfully!")
    print("=" * 70)


def test_step_by_step():
    """Test each component individually."""
    
    print("\n\n" + "=" * 70)
    print("Step-by-Step Component Test")
    print("=" * 70)
    
    model = MonoDETR(num_classes=3, num_queries=100, hidden_dim=256)
    model.eval()
    
    images = torch.randn(2, 3, 384, 512)
    
    print("\nStep 1: Feature Extraction")
    print("-" * 70)
    visual_features = model.feature_extractor(images)
    print(f"  Input:  {images.shape}")
    print(f"  Output: {visual_features.shape}")
    print(f"  → Extracts CNN features from image (ResNet50 backbone)")
    
    print("\nStep 2: Depth Prediction")
    print("-" * 70)
    depth_map = model.depth_predictor(visual_features)
    print(f"  Input:  {visual_features.shape}")
    print(f"  Output: {depth_map.shape}")
    print(f"  → Predicts monocular depth map from visual features")
    
    print("\nStep 3: Visual Encoding")
    print("-" * 70)
    visual_encoded = model.visual_encoder(visual_features, num_queries=100)
    print(f"  Input:  {visual_features.shape}")
    print(f"  Output: {visual_encoded.shape}")
    print(f"  → Encodes visual features with transformer self-attention")
    
    print("\nStep 4: Depth Encoding")
    print("-" * 70)
    depth_encoded = model.depth_encoder(depth_map, num_queries=100)
    print(f"  Input:  {depth_map.shape}")
    print(f"  Output: {depth_encoded.shape}")
    print(f"  → Encodes depth map with transformer self-attention")
    
    print("\nStep 5: Visual-Depth Fusion")
    print("-" * 70)
    fused_features = model.visual_depth_decoder(visual_encoded, depth_encoded)
    print(f"  Visual Input: {visual_encoded.shape}")
    print(f"  Depth Input:  {depth_encoded.shape}")
    print(f"  Output:       {fused_features.shape}")
    print(f"  → Fuses visual and depth with learnable object queries")
    
    print("\nStep 6: Prediction Heads")
    print("-" * 70)
    predictions = model.decoder_heads(fused_features)
    print(f"  Input:  {fused_features.shape}")
    for key, val in predictions.items():
        print(f"  Output ({key}): {val.shape}")
    print(f"  → Generates 2D/3D boxes and classification logits")
    
    print("\n" + "=" * 70)


def test_with_visualization():
    """Test with some analysis of the outputs."""
    
    print("\n\n" + "=" * 70)
    print("Output Analysis Test")
    print("=" * 70)
    
    model = MonoDETR(num_classes=3, num_queries=100, hidden_dim=256)
    model.eval()
    
    images = torch.randn(2, 3, 384, 512)
    
    with torch.no_grad():
        outputs = model(images)
    
    print("\nAnalyzing predictions from 1st sample in batch:\n")
    
    # 2D boxes
    boxes_2d = outputs['pred_boxes_2d'][0]  # First sample
    print(f"2D Boxes: {boxes_2d.shape}")
    print(f"  First box (normalized coords): {boxes_2d[0].numpy()}")
    print(f"  Range: x_min, y_min, x_max, y_max (all normalized to [0, 1])\n")
    
    # 3D boxes
    boxes_3d = outputs['pred_boxes_3d'][0]
    print(f"3D Boxes: {boxes_3d.shape}")
    print(f"  First box (camera coords): {boxes_3d[0].numpy()}")
    print(f"  Format: [center_x, center_y, center_z, length, width, height, rotation]\n")
    
    # Logits
    logits = outputs['pred_logits'][0]
    print(f"Classification Logits: {logits.shape}")
    print(f"  First query logits (3 classes): {logits[0].numpy()}")
    print(f"  Classes: 0=Car, 1=Pedestrian, 2=Cyclist\n")
    
    # Depth
    depth = outputs['pred_depth']
    print(f"Depth Map: {depth.shape}")
    print(f"  Min depth: {depth.min():.3f}")
    print(f"  Max depth: {depth.max():.3f}")
    print(f"  Mean depth: {depth.mean():.3f}\n")
    
    print("=" * 70)


if __name__ == '__main__':
    # Run all tests
    test_end_to_end()
    test_step_by_step()
    test_with_visualization()
    
    print("\n✓ All tests completed!")
