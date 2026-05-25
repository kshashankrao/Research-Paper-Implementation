"""
RegressionLoss: 3D bounding box regression losses.

This module computes regression losses for 3D bounding box predictions.
Includes L1 loss for general shape and GIoU loss for better box alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_ops import generalized_box3d_iou, get_src_permutation_idx


class RegressionLoss(nn.Module):
    """
    3D bounding box regression loss combining L1 and GIoU losses.
    
    The 3D box is represented as: [cx, cy, cz, length, width, height, rotation]
    - cx, cy, cz: 3D center in camera coordinates
    - length, width, height: Box dimensions
    - rotation: Rotation angle around vertical axis
    
    Args:
        weight_l1 (float): Weight of L1 loss component. Default: 1.0
        weight_giou (float): Weight of GIoU loss component. Default: 1.0
    """
    
    def __init__(self, weight_l1: float = 1.0, weight_giou: float = 1.0):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou
    
    def forward(self, pred_boxes_3d: torch.Tensor, gt_boxes_3d: torch.Tensor,
                num_boxes: int) -> dict:
        """
        Compute 3D box regression loss.
        
        Args:
            pred_boxes_3d (torch.Tensor): Predicted 3D boxes [num_matched, 7]
                                         Format: [cx, cy, cz, l, w, h, rotation]
            gt_boxes_3d (torch.Tensor): Ground truth 3D boxes [num_matched, 7]
            num_boxes (int): Total number of ground truth boxes in the batch
                           Used for normalization
        
        Returns:
            dict: Dictionary containing individual loss components:
                - 'loss_bbox_3d': L1 regression loss
                - 'loss_giou_3d': GIoU loss
        
        Example:
            >>> loss_fn = RegressionLoss()
            >>> pred = torch.randn(50, 7)  # 50 matched predictions
            >>> gt = torch.randn(50, 7)
            >>> losses = loss_fn(pred, gt, num_boxes=100)
            >>> total_loss = losses['loss_bbox_3d'] + losses['loss_giou_3d']
        """
        losses = {}
        
        # L1 loss on 3D box parameters
        # Penalizes absolute differences in center, dimensions, and rotation
        loss_l1 = F.l1_loss(pred_boxes_3d, gt_boxes_3d, reduction='none')
        
        # Normalize by number of boxes (average per box across batch)
        losses['loss_bbox_3d'] = self.weight_l1 * loss_l1.sum() / max(num_boxes, 1)
        
        # GIoU loss for better box alignment
        # Encourages boxes to be well-aligned in 3D space
        giou_3d = generalized_box3d_iou(pred_boxes_3d, gt_boxes_3d)
        loss_giou = 1 - giou_3d  # Convert to loss (lower is better)
        
        losses['loss_giou_3d'] = self.weight_giou * loss_giou.sum() / max(num_boxes, 1)
        
        return losses


class Box2dRegressionLoss(nn.Module):
    """
    2D bounding box regression loss for monocular detection.
    
    The 2D box is represented in normalized [0, 1] format: [x_min, y_min, x_max, y_max]
    
    Args:
        weight_l1 (float): Weight of L1 loss. Default: 1.0
    """
    
    def __init__(self, weight_l1: float = 1.0):
        super().__init__()
        self.weight_l1 = weight_l1
    
    def forward(self, pred_boxes_2d: torch.Tensor, gt_boxes_2d: torch.Tensor,
                num_boxes: int) -> dict:
        """
        Compute 2D box regression loss.
        
        Args:
            pred_boxes_2d (torch.Tensor): Predicted 2D boxes [num_matched, 4]
            gt_boxes_2d (torch.Tensor): Ground truth 2D boxes [num_matched, 4]
            num_boxes (int): Total number of ground truth boxes (for normalization)
        
        Returns:
            dict: Dictionary with loss component 'loss_bbox_2d'
        """
        losses = {}
        
        # L1 loss on 2D box coordinates
        loss_l1 = F.l1_loss(pred_boxes_2d, gt_boxes_2d, reduction='none')
        
        losses['loss_bbox_2d'] = self.weight_l1 * loss_l1.sum() / max(num_boxes, 1)
        
        return losses


class CombinedRegressionLoss(nn.Module):
    """
    Combined regression loss for both 2D and 3D boxes.
    
    Jointly optimizes 2D projection consistency and 3D box geometry.
    
    Args:
        weight_2d (float): Weight of 2D loss. Default: 0.5
        weight_3d (float): Weight of 3D loss. Default: 1.0
    """
    
    def __init__(self, weight_2d: float = 0.5, weight_3d: float = 1.0):
        super().__init__()
        self.weight_2d = weight_2d
        self.weight_3d = weight_3d
        
        self.loss_2d = Box2dRegressionLoss()
        self.loss_3d = RegressionLoss()
    
    def forward(self, pred_boxes_2d: torch.Tensor, pred_boxes_3d: torch.Tensor,
                gt_boxes_2d: torch.Tensor, gt_boxes_3d: torch.Tensor,
                num_boxes: int) -> dict:
        """
        Compute combined 2D and 3D regression loss.
        
        Args:
            pred_boxes_2d (torch.Tensor): Predicted 2D boxes [num_matched, 4]
            pred_boxes_3d (torch.Tensor): Predicted 3D boxes [num_matched, 7]
            gt_boxes_2d (torch.Tensor): Ground truth 2D boxes [num_matched, 4]
            gt_boxes_3d (torch.Tensor): Ground truth 3D boxes [num_matched, 7]
            num_boxes (int): Total number of ground truth boxes
        
        Returns:
            dict: Combined loss dictionary
        """
        losses = {}
        
        # Compute 2D loss
        losses_2d = self.loss_2d(pred_boxes_2d, gt_boxes_2d, num_boxes)
        for key, val in losses_2d.items():
            losses[key] = self.weight_2d * val
        
        # Compute 3D loss
        losses_3d = self.loss_3d(pred_boxes_3d, gt_boxes_3d, num_boxes)
        for key, val in losses_3d.items():
            losses[key] = self.weight_3d * val
        
        return losses
