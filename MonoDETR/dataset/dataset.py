"""
Dataset implementations for MonoDETR training.

This module provides dataset classes for loading and preparing data for MonoDETR.
Replace DummyDataset with actual dataset implementations for real training.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple
from pathlib import Path
import numpy as np


class DummyDataset(Dataset):
    """
    Dummy dataset for demonstration and testing purposes.
    
    Generates random images, depth maps, and object annotations.
    Use this for testing the training pipeline before implementing real datasets.
    
    Args:
        num_samples (int): Number of samples in the dataset. Default: 100
        img_size (tuple): Image size as (height, width). Default: (384, 512)
        num_classes (int): Number of object classes. Default: 3
        max_objects (int): Maximum number of objects per image. Default: 20
    
    Returns:
        Tuple containing:
        - image: torch.Tensor of shape [3, height, width]
        - depth: torch.Tensor of shape [1, height, width]
        - targets: Dict with 'labels', 'boxes_2d', 'boxes_3d'
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        img_size: Tuple = (384, 512),
        num_classes: int = 3,
        max_objects: int = 20
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_objects = max_objects
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get sample at index.
        
        Args:
            idx (int): Sample index
        
        Returns:
            Tuple:
            - image: [3, height, width] - Random RGB image
            - depth: [1, height, width] - Random depth map (values in [0, 1])
            - targets: Dict containing ground truth annotations
                - 'labels': [num_objects] - Class labels (0 to num_classes-1)
                - 'boxes_2d': [num_objects, 4] - Normalized 2D boxes [x_min, y_min, x_max, y_max]
                - 'boxes_3d': [num_objects, 7] - 3D boxes [cx, cy, cz, l, w, h, rotation]
        """
        # Generate random image
        # Shape: [3, height, width]
        image = torch.randn(3, *self.img_size)
        
        # Generate random depth map
        # Values in [0, 1] representing normalized depth
        # Shape: [1, height, width]
        depth = torch.rand(1, *self.img_size)
        
        # Generate random number of objects (1 to max_objects)
        num_objects = torch.randint(1, self.max_objects, (1,)).item()
        
        # Create target annotations
        targets = {
            # Class labels: integers from 0 to num_classes-1
            'labels': torch.randint(0, self.num_classes, (num_objects,)),
            
            # 2D bounding boxes: normalized coordinates [x_min, y_min, x_max, y_max]
            # Values in [0, 1] representing normalized image coordinates
            'boxes_2d': torch.rand(num_objects, 4),
            
            # 3D bounding boxes: [cx, cy, cz, length, width, height, rotation]
            # - cx, cy, cz: 3D center in camera coordinates
            # - length, width, height: box dimensions (can be any scale)
            # - rotation: rotation angle around vertical axis
            'boxes_3d': torch.randn(num_objects, 7),
        }
        
        return image, depth, targets


class KITTIDataset(Dataset):
    """
    KITTI 3D Object Detection Dataset.
    
    Reads KITTI dataset format for 3D object detection.
    Reference: http://www.cvlibs.net/datasets/kitti/
    
    Expected directory structure:
        kitti_root/
        ├── image_2/           # Left color images
        ├── depth_maps/        # Depth maps (optional)
        ├── calib/             # Calibration files
        └── label_2/           # 3D object labels
    
    Args:
        root_dir (str): Root directory of KITTI dataset
        split (str): 'train', 'val', or 'test'. Default: 'train'
        use_depth (bool): Whether to load depth maps. Default: False
    """
    
    def __init__(self, root_dir: str, split: str = 'train', use_depth: bool = False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.use_depth = use_depth
        
        # Build sample list based on split
        self.samples = self._build_sample_list()
    
    def _build_sample_list(self):
        """Build list of sample indices for the split."""
        # TODO: Implement KITTI split reading
        # This should read train.txt, val.txt, or test.txt from KITTI
        raise NotImplementedError("KITTI dataset loading not implemented")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get sample at index.
        
        Returns:
            Tuple:
            - image: [3, height, width]
            - depth: [1, height, width] or None if use_depth=False
            - targets: Dict with annotations
        """
        # TODO: Implement KITTI sample loading
        raise NotImplementedError("KITTI dataset loading not implemented")


class NuScenesDataset(Dataset):
    """
    nuScenes 3D Object Detection Dataset.
    
    Reads nuScenes dataset format for 3D object detection.
    Reference: https://www.nuscenes.org/
    
    Args:
        root_dir (str): Root directory of nuScenes dataset
        split (str): 'train', 'val', or 'test'. Default: 'train'
        version (str): Dataset version. Default: 'v1.0-trainval'
    """
    
    def __init__(self, root_dir: str, split: str = 'train', version: str = 'v1.0-trainval'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.version = version
        
        # TODO: Load nuScenes database
        self.samples = []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get sample at index."""
        # TODO: Implement nuScenes sample loading
        raise NotImplementedError("nuScenes dataset loading not implemented")


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles variable number of objects per sample by padding or using lists.
    
    Args:
        batch: List of (image, depth, targets) tuples
    
    Returns:
        Tuple:
        - images: [batch_size, 3, height, width]
        - depths: [batch_size, 1, height, width]
        - targets: List of target dicts (one per sample)
    """
    images = []
    depths = []
    targets_list = []
    
    for image, depth, targets in batch:
        images.append(image)
        depths.append(depth)
        targets_list.append(targets)
    
    # Stack images and depths
    images = torch.stack(images, dim=0)
    depths = torch.stack(depths, dim=0)
    
    return images, depths, targets_list


if __name__ == '__main__':
    """Quick test of dataset implementations."""
    print("Testing DummyDataset...")
    
    # Create dummy dataset
    dataset = DummyDataset(num_samples=10, img_size=(384, 512), num_classes=3)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Get sample
    image, depth, targets = dataset[0]
    print(f"✓ Sample 0:")
    print(f"  - Image shape: {image.shape}")
    print(f"  - Depth shape: {depth.shape}")
    print(f"  - Labels shape: {targets['labels'].shape}")
    print(f"  - 2D boxes shape: {targets['boxes_2d'].shape}")
    print(f"  - 3D boxes shape: {targets['boxes_3d'].shape}")
    
    print("\n✓ Dataset test passed!")
