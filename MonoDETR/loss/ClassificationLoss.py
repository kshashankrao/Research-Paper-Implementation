"""
ClassificationLoss: Cross-entropy loss for object classification.

This module computes the classification loss for 3D object detection.
It handles both positive (objects present) and negative (background) predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Multi-class classification loss with focal loss support.
    
    Uses cross-entropy loss for object classification. Supports focal loss
    to handle class imbalance (more difficult negatives weighted higher).
    
    Args:
        num_classes (int): Number of object classes. Default: 3
        focal_alpha (float): Focal loss alpha parameter. Default: 0.25
                           If None, standard cross-entropy is used. Default: None
        focal_gamma (float): Focal loss gamma parameter (focusing parameter). Default: 2.0
                           Higher gamma down-weights easy examples more.
    """
    
    def __init__(self, num_classes: int = 3, focal_alpha: float = None,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_classes_with_bg = num_classes + 1  # +1 for background/no-object class
    
    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            pred_logits (torch.Tensor): Predicted class logits [Batch, num_queries, num_classes + 1]
            gt_labels (torch.Tensor): Ground truth class labels [Batch, num_queries]
                                     Labels should be in range [0, num_classes]
                                     where 0 is background/no-object
        
        Returns:
            torch.Tensor: Scalar classification loss
        
        Example:
            >>> loss_fn = ClassificationLoss(num_classes=3)
            >>> pred_logits = torch.randn(2, 100, 4)  # 2 batches, 100 queries, 4 classes (3 + background)
            >>> gt_labels = torch.randint(0, 4, (2, 100))
            >>> loss = loss_fn(pred_logits, gt_labels)
        """
        # Flatten predictions and labels for easier computation
        # Shape: [Batch * num_queries, num_classes + 1]
        pred_flat = pred_logits.reshape(-1, self.num_classes_with_bg)
        
        # Shape: [Batch * num_queries]
        labels_flat = gt_labels.reshape(-1)
        
        # Compute cross-entropy loss
        # Shape: [Batch * num_queries]
        ce_loss = F.cross_entropy(pred_flat, labels_flat, reduction='none')
        
        if self.focal_alpha is not None:
            # Apply focal loss for handling class imbalance
            # Focal loss: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
            # where pt is the probability of the correct class
            
            # Get probabilities
            probs = F.softmax(pred_flat, dim=-1)
            
            # Get probability of ground truth class for each sample
            # Shape: [Batch * num_queries]
            p_t = probs.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            
            # Compute focal loss weight: (1 - p_t)^gamma
            focal_weight = (1 - p_t) ** self.focal_gamma
            
            # Apply alpha weighting and focal weight
            ce_loss = self.focal_alpha * focal_weight * ce_loss
        
        # Return mean loss
        return ce_loss.mean()


class WeightedClassificationLoss(nn.Module):
    """
    Classification loss with per-class weighting.
    
    Allows specifying different loss weights for different classes,
    useful when classes are imbalanced in the dataset.
    
    Args:
        num_classes (int): Number of object classes. Default: 3
        class_weights (torch.Tensor): Weight for each class [num_classes + 1].
                                     If None, equal weights used. Default: None
    """
    
    def __init__(self, num_classes: int = 3, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_with_bg = num_classes + 1
        
        if class_weights is None:
            class_weights = torch.ones(self.num_classes_with_bg)
        
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, pred_logits: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted classification loss.
        
        Args:
            pred_logits (torch.Tensor): Predicted class logits [Batch, num_queries, num_classes + 1]
            gt_labels (torch.Tensor): Ground truth class labels [Batch, num_queries]
        
        Returns:
            torch.Tensor: Scalar weighted classification loss
        """
        # Flatten predictions and labels
        pred_flat = pred_logits.reshape(-1, self.num_classes_with_bg)
        labels_flat = gt_labels.reshape(-1)
        
        # Compute weighted cross-entropy loss
        loss = F.cross_entropy(
            pred_flat,
            labels_flat,
            weight=self.class_weights,
            reduction='mean'
        )
        
        return loss
