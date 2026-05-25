"""
Dataset module for MonoDETR.

Provides dataset implementations for training and evaluation:
- DummyDataset: Random data for testing
- KITTIDataset: KITTI 3D object detection dataset
- NuScenesDataset: nuScenes 3D object detection dataset
"""

from .dataset import (
    DummyDataset,
    KITTIDataset,
    NuScenesDataset,
    collate_fn,
)

__all__ = [
    'DummyDataset',
    'KITTIDataset',
    'NuScenesDataset',
    'collate_fn',
]
