"""
Loss: Main loss computation module for MonoDETR.

This module wraps all loss components (depth, classification, regression)
and provides a unified interface for computing the total training loss.

The training loss is a weighted combination of:
1. Depth prediction loss - ensures the model learns monocular depth
2. Classification loss - ensures correct object class prediction
3. 2D Regression loss - ensures 2D projection accuracy
4. 3D Regression loss - ensures 3D box accuracy
"""

import torch
import torch.nn as nn
from .Hungarian3dBboxMatching import Hungarian3dBboxMatching
from .DepthLoss import DepthLoss, DepthVariationLoss
from .ClassificationLoss import ClassificationLoss, WeightedClassificationLoss
from .RegressionLoss import Box2dRegressionLoss, RegressionLoss, CombinedRegressionLoss
from .box_ops import get_src_permutation_idx, box3d_to_2d


class Loss(nn.Module):
    """
    Complete loss computation for MonoDETR.
    
    Combines multiple loss components with learnable weights for joint optimization.
    Uses Hungarian matching to assign predictions to ground truth before computing losses.
    
    Args:
        num_classes (int): Number of object classes. Default: 3
        weight_depth (float): Weight of depth prediction loss. Default: 1.0
        weight_depth_smooth (float): Weight of depth smoothness loss. Default: 0.1
        weight_class (float): Weight of classification loss. Default: 1.0
        weight_bbox_2d (float): Weight of 2D box regression loss. Default: 1.0
        weight_bbox_3d (float): Weight of 3D box regression loss. Default: 2.0
        use_focal_loss (bool): Whether to use focal loss for classification. Default: False
    
    Example:
        >>> loss_fn = Loss(num_classes=3)
        >>> outputs = model(images)  # Model forward pass
        >>> depth_gt = load_depth_maps()  # Load GT depth
        >>> losses = loss_fn(outputs, targets, depth_gt)
        >>> total_loss = losses['loss_total']
    """
    
    def __init__(self, num_classes: int = 3, weight_depth: float = 1.0,
                 weight_depth_smooth: float = 0.1, weight_class: float = 1.0,
                 weight_bbox_2d: float = 1.0, weight_bbox_3d: float = 2.0,
                 use_focal_loss: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Loss weights for each component
        self.weight_depth = weight_depth
        self.weight_depth_smooth = weight_depth_smooth
        self.weight_class = weight_class
        self.weight_bbox_2d = weight_bbox_2d
        self.weight_bbox_3d = weight_bbox_3d
        
        # Matching module: assigns predictions to ground truth
        self.matcher = Hungarian3dBboxMatching(
            cost_class=1.0,
            cost_3dbbox=5.0,
            cost_giou=2.0
        )
        
        # Individual loss modules
        self.depth_loss = DepthLoss(weight_l1=0.1, weight_ssim=0.9)
        self.depth_smooth_loss = DepthVariationLoss()
        
        if use_focal_loss:
            self.class_loss = ClassificationLoss(
                num_classes=num_classes,
                focal_alpha=0.25,
                focal_gamma=2.0
            )
        else:
            self.class_loss = ClassificationLoss(num_classes=num_classes)
        
        self.bbox_2d_loss = Box2dRegressionLoss(weight_l1=1.0)
        self.bbox_3d_loss = RegressionLoss(weight_l1=1.0, weight_giou=1.0)
    
    def forward(self, outputs: dict, targets: list, depth_gt: torch.Tensor = None,
                depth_mask: torch.Tensor = None) -> dict:
        """
        Compute total loss for training.
        
        Args:
            outputs (dict): Model predictions containing:
                - 'pred_logits': Classification logits [Batch, num_queries, num_classes + 1]
                - 'pred_boxes_2d': 2D boxes [Batch, num_queries, 4]
                - 'pred_boxes_3d': 3D boxes [Batch, num_queries, 7]
                - 'pred_depth': Predicted depth [Batch, 1, H, W]
            targets (list): Ground truth list of dicts, each containing:
                - 'labels': Object class labels [num_objects]
                - 'boxes_2d': Ground truth 2D boxes [num_objects, 4]
                - 'boxes_3d': Ground truth 3D boxes [num_objects, 7]
            depth_gt (torch.Tensor): Ground truth depth maps [Batch, 1, H, W]. Optional
            depth_mask (torch.Tensor): Valid depth mask [Batch, 1, H, W]. Optional
        
        Returns:
            dict: Loss dictionary containing:
                - 'loss_total': Weighted sum of all losses (scalar)
                - 'loss_depth': Depth prediction loss
                - 'loss_depth_smooth': Depth smoothness regularization
                - 'loss_class': Classification loss
                - 'loss_bbox_2d': 2D box regression loss
                - 'loss_bbox_3d': 3D box regression loss
                - 'loss_giou_3d': 3D GIoU loss
        
        Example:
            >>> outputs = {
            ...     'pred_logits': torch.randn(2, 100, 4),
            ...     'pred_boxes_2d': torch.randn(2, 100, 4),
            ...     'pred_boxes_3d': torch.randn(2, 100, 7),
            ...     'pred_depth': torch.randn(2, 1, 384, 512)
            ... }
            >>> targets = [
            ...     {
            ...         'labels': torch.tensor([0, 2]),
            ...         'boxes_2d': torch.randn(2, 4),
            ...         'boxes_3d': torch.randn(2, 7)
            ...     },
            ...     {...}
            ... ]
            >>> depth_gt = torch.randn(2, 1, 384, 512)
            >>> losses = loss_fn(outputs, targets, depth_gt)
        """
        losses = {}
        
        # Step 1: Match predictions to ground truth using Hungarian algorithm
        indices = self.matcher(outputs, targets)
        
        # Get total number of boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)  # Avoid division by zero
        
        # Step 2: Compute classification loss
        # Create ground truth labels for all queries
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        gt_labels = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # Background class index
            dtype=torch.long,
            device=outputs["pred_logits"].device
        )
        
        # Fill in ground truth labels for matched predictions
        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            gt_labels[batch_idx, pred_idx] = targets[batch_idx]["labels"][tgt_idx]
        
        loss_class = self.class_loss(outputs["pred_logits"], gt_labels)
        losses['loss_class'] = self.weight_class * loss_class
        
        # Step 3: Compute 2D and 3D box regression losses
        # Extract matched predictions and targets
        idx_batch, idx_pred = get_src_permutation_idx(indices)
        
        # Matched predictions
        pred_boxes_2d_matched = outputs["pred_boxes_2d"][idx_batch, idx_pred]
        pred_boxes_3d_matched = outputs["pred_boxes_3d"][idx_batch, idx_pred]
        
        # Matched ground truth boxes
        gt_boxes_2d_matched = torch.cat([
            targets[i]["boxes_2d"][j] for i, (_, j) in enumerate(indices)
        ])
        gt_boxes_3d_matched = torch.cat([
            targets[i]["boxes_3d"][j] for i, (_, j) in enumerate(indices)
        ])
        
        # Compute 2D box loss
        loss_2d_dict = self.bbox_2d_loss(pred_boxes_2d_matched, gt_boxes_2d_matched, num_boxes)
        losses['loss_bbox_2d'] = self.weight_bbox_2d * loss_2d_dict['loss_bbox_2d']
        
        # Compute 3D box loss
        loss_3d_dict = self.bbox_3d_loss(pred_boxes_3d_matched, gt_boxes_3d_matched, num_boxes)
        losses['loss_bbox_3d'] = self.weight_bbox_3d * loss_3d_dict['loss_bbox_3d']
        losses['loss_giou_3d'] = self.weight_bbox_3d * loss_3d_dict['loss_giou_3d']
        
        # Step 4: Compute depth prediction loss (if ground truth depth is provided)
        if depth_gt is not None and 'pred_depth' in outputs:
            loss_depth = self.depth_loss(
                outputs['pred_depth'],
                depth_gt,
                mask=depth_mask
            )
            losses['loss_depth'] = self.weight_depth * loss_depth
            
            # Add depth smoothness regularization
            loss_depth_smooth = self.depth_smooth_loss(outputs['pred_depth'])
            losses['loss_depth_smooth'] = self.weight_depth_smooth * loss_depth_smooth
        else:
            losses['loss_depth'] = torch.tensor(0.0, device=outputs["pred_logits"].device)
            losses['loss_depth_smooth'] = torch.tensor(0.0, device=outputs["pred_logits"].device)
        
        # Step 5: Compute total weighted loss
        total_loss = (
            losses['loss_class'] +
            losses['loss_bbox_2d'] +
            losses['loss_bbox_3d'] +
            losses['loss_giou_3d'] +
            losses['loss_depth'] +
            losses['loss_depth_smooth']
        )
        losses['loss_total'] = total_loss
        
        return losses
