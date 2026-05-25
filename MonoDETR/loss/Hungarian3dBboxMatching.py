"""
Hungarian3dBboxMatching: Optimal assignment between predicted and ground truth 3D boxes.

This module implements the Hungarian algorithm for matching predicted 3D bounding boxes
to ground truth boxes using a cost matrix that combines classification, 3D box regression,
and other metrics.

The matching is essential for computing losses - it determines which predictions
correspond to which ground truth boxes.
"""

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from .box_ops import generalized_box3d_iou


class Hungarian3dBboxMatching(nn.Module):
    """
    Hungarian algorithm for optimal matching between predicted and ground truth 3D boxes.
    
    The matching cost combines multiple factors:
    1. Classification cost: -probability of ground truth class
    2. 3D box regression cost: L1 distance between predicted and GT 3D boxes
    3. 3D IoU cost: Negative GIoU of predicted and GT boxes
    
    Args:
        cost_class (float): Relative weight of classification cost. Default: 1.0
        cost_3dbbox (float): Relative weight of 3D box regression cost. Default: 5.0
        cost_giou (float): Relative weight of GIoU cost. Default: 2.0
    
    Example:
        >>> matcher = Hungarian3dBboxMatching(cost_class=1, cost_3dbbox=5, cost_giou=2)
        >>> indices = matcher(outputs, targets)
        >>> # indices contains matched (pred_idx, target_idx) pairs
    """
    
    def __init__(self, cost_class: float = 1.0, cost_3dbbox: float = 5.0, 
                 cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dbbox = cost_3dbbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Match predictions to ground truth using Hungarian algorithm.
        
        Args:
            outputs (dict): Model predictions containing:
                - 'pred_logits': Classification logits [Batch, num_queries, num_classes + 1]
                - 'pred_boxes_3d': 3D box predictions [Batch, num_queries, 7]
            targets (list): List of ground truth dicts, each containing:
                - 'labels': Object class labels [num_targets]
                - 'boxes_3d': 3D ground truth boxes [num_targets, 7]
        
        Returns:
            list: List of (query_indices, target_indices) tuples for each batch element
        
        Example:
            >>> outputs = {
            ...     'pred_logits': torch.randn(2, 100, 4),  # 2 batches, 100 queries, 4 classes
            ...     'pred_boxes_3d': torch.randn(2, 100, 7)
            ... }
            >>> targets = [
            ...     {'labels': torch.tensor([0, 2]), 'boxes_3d': torch.randn(2, 7)},
            ...     {'labels': torch.tensor([1]), 'boxes_3d': torch.randn(1, 7)}
            ... ]
            >>> matcher = Hungarian3dBboxMatching()
            >>> indices = matcher(outputs, targets)
            >>> # indices = [(query_idx_batch0, target_idx_batch0), (query_idx_batch1, target_idx_batch1)]
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten batch and query dimensions for cost computation
        # Shape: [Batch * num_queries, num_classes + 1]
        pred_probs = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        
        # Shape: [Batch * num_queries, 7]
        pred_boxes_3d = outputs["pred_boxes_3d"].flatten(0, 1)
        
        # Concatenate all ground truth across batch
        tgt_labels = torch.cat([v["labels"] for v in targets])
        tgt_boxes_3d = torch.cat([v["boxes_3d"] for v in targets])
        
        # Compute classification cost: negative probability of ground truth class
        # For each predicted box, compute -prob[ground_truth_class]
        # Shape: [Batch * num_queries, num_targets]
        cost_class = -pred_probs[:, tgt_labels]
        
        # Compute L1 regression cost between 3D boxes
        # For each predicted 3D box, compute L1 distance to all ground truth boxes
        # Shape: [Batch * num_queries, num_targets]
        cost_3dbbox = torch.cdist(pred_boxes_3d, tgt_boxes_3d, p=1)
        
        # Compute IoU cost: negative GIoU
        # Shape: [Batch * num_queries, num_targets]
        cost_giou = -generalized_box3d_iou(
            pred_boxes_3d.unsqueeze(1).expand(-1, tgt_boxes_3d.shape[0], -1),
            tgt_boxes_3d.unsqueeze(0).expand(pred_boxes_3d.shape[0], -1, -1)
        )
        
        # Combine costs with learned weights
        # Final cost matrix: [Batch * num_queries, num_targets]
        cost_matrix = (
            self.cost_class * cost_class +
            self.cost_3dbbox * cost_3dbbox +
            self.cost_giou * cost_giou
        )
        
        # Reshape to per-batch: [Batch, num_queries, num_targets]
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()
        
        # Get sizes of ground truth for each batch element
        sizes = [len(v["boxes_3d"]) for v in targets]
        
        # Apply Hungarian algorithm to each batch element
        indices = []
        for i, (cost, size) in enumerate(zip(cost_matrix, sizes)):
            # Split cost matrix for this batch element
            # cost shape: [num_queries, num_targets_in_batch]
            cost_batch = cost[:, :size]
            
            # Apply Hungarian algorithm
            pred_idx, tgt_idx = linear_sum_assignment(cost_batch.numpy())
            
            # Convert to tensors
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64)
            ))
        
        return indices
