"""
Loss utilities: Box operations and IOU computations for 3D bounding boxes.

This module provides helper functions for 3D box computations needed in loss calculations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def generalized_box3d_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU for 3D bounding boxes.
    
    This is an approximation using 2D projections of the 3D boxes.
    In practice, 3D IoU involves computing intersection and union volumes,
    but this 2D approximation is more computationally efficient.
    
    Args:
        boxes1 (torch.Tensor): 3D boxes of shape [..., 7]
                              Format: [cx, cy, cz, l, w, h, rotation]
        boxes2 (torch.Tensor): 3D boxes of shape [..., 7]
    
    Returns:
        torch.Tensor: GIoU values of shape [...]
    
    Note:
        For now, we use 2D projection (ignoring z coordinate) as a proxy.
        A full 3D implementation would compute volume intersection.
    """
    # Extract 2D projection (cx, cy, w, h, ignore z)
    # This is a simplified version using 2D boxes from 3D boxes
    
    # Get box centers and dimensions
    centers1 = boxes1[..., :2]  # [cx, cy]
    dims1 = boxes1[..., 3:5]    # [l, w] (length and width)
    
    centers2 = boxes2[..., :2]
    dims2 = boxes2[..., 3:5]
    
    # Convert to [x_min, y_min, x_max, y_max] format for 2D boxes
    boxes1_2d = torch.cat([
        centers1 - dims1 / 2,
        centers1 + dims1 / 2
    ], dim=-1)
    
    boxes2_2d = torch.cat([
        centers2 - dims2 / 2,
        centers2 + dims2 / 2
    ], dim=-1)
    
    return generalized_box_iou_2d(boxes1_2d, boxes2_2d)


def generalized_box_iou_2d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU for 2D boxes in [x_min, y_min, x_max, y_max] format.
    
    Args:
        boxes1 (torch.Tensor): Boxes of shape [..., 4]
        boxes2 (torch.Tensor): Boxes of shape [..., 4]
    
    Returns:
        torch.Tensor: GIoU values
    """
    # Compute areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Compute intersection
    inter_min = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_max = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_size[..., 0] * inter_size[..., 1]
    
    # Compute union
    union_area = area1 + area2 - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-8)
    
    # Compute enclosing box for GIoU
    enclosing_min = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclosing_max = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_size = enclosing_max - enclosing_min
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    
    # Compute GIoU
    giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
    
    return giou


def box3d_to_2d(boxes_3d: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D boxes to 2D projection for matching.
    
    Args:
        boxes_3d (torch.Tensor): 3D boxes of shape [..., 7]
                               Format: [cx, cy, cz, l, w, h, rotation]
    
    Returns:
        torch.Tensor: 2D boxes of shape [..., 4] in [x_min, y_min, x_max, y_max]
    """
    centers = boxes_3d[..., :2]  # [cx, cy]
    dims = boxes_3d[..., 3:5]    # [l, w]
    
    # Convert to [x_min, y_min, x_max, y_max]
    boxes_2d = torch.cat([
        centers - dims / 2,
        centers + dims / 2
    ], dim=-1)
    
    return boxes_2d


def get_src_permutation_idx(indices):
    """
    Extract the permutation indices from matched pairs.
    
    Args:
        indices: List of (batch_idx, target_idx) tuples from Hungarian matching
    
    Returns:
        Tuple of (batch_idx, query_idx) for indexing predictions
    """
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
