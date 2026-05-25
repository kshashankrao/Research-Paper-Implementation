"""
DepthLoss: Loss for monocular depth prediction supervision.

This module computes the loss for the depth predictor network. The depth prediction
is trained with ground truth depth supervision to improve the model's understanding
of scene geometry and 3D object locations.

Supports both dense depth maps and sparse depth supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    """
    Depth prediction loss combining multiple objectives.
    
    Combines L1 loss (for general shape) and structural similarity loss (SSIM)
    to encourage both pixel-level accuracy and perceptually correct depth maps.
    
    Args:
        weight_l1 (float): Weight of L1 loss. Default: 0.1
        weight_ssim (float): Weight of SSIM loss. Default: 0.9
        mask_invalid (bool): Whether to mask out invalid depth values. Default: True
    """
    
    def __init__(self, weight_l1: float = 0.1, weight_ssim: float = 0.9, 
                 mask_invalid: bool = True):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_ssim = weight_ssim
        self.mask_invalid = mask_invalid
    
    def compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, 
                     window_size: int = 11) -> torch.Tensor:
        """
        Compute Structural Similarity Index (SSIM) between two images/depth maps.
        
        SSIM measures structural similarity which often correlates better with
        perceptual quality than simple pixel-level differences.
        
        Args:
            img1 (torch.Tensor): First image [Batch, C, H, W]
            img2 (torch.Tensor): Second image [Batch, C, H, W]
            window_size (int): Size of Gaussian window. Default: 11
        
        Returns:
            torch.Tensor: Mean SSIM value (scalar)
        """
        # Create Gaussian kernel for SSIM computation
        sigma = 1.5
        kernel_range = torch.arange(
            -(window_size // 2), (window_size // 2) + 1,
            dtype=torch.float32, device=img1.device
        )
        kernel = torch.exp(-(kernel_range ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = kernel.view(-1, 1) @ kernel.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, window_size, window_size)
        
        # Compute local means
        mu1 = F.conv2d(img1, kernel_2d, padding=window_size // 2)
        mu2 = F.conv2d(img2, kernel_2d, padding=window_size // 2)
        
        # Compute local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 ** 2, kernel_2d, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, kernel_2d, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=window_size // 2) - mu1_mu2
        
        # SSIM constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute SSIM map
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return mean SSIM
        return ssim_map.mean()
    
    def forward(self, pred_depth: torch.Tensor, gt_depth: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute depth prediction loss.
        
        Args:
            pred_depth (torch.Tensor): Predicted depth maps [Batch, 1, Height, Width]
            gt_depth (torch.Tensor): Ground truth depth maps [Batch, 1, Height, Width]
            mask (torch.Tensor): Valid depth mask [Batch, 1, Height, Width]. 
                               If None, all pixels are considered valid. Default: None
        
        Returns:
            torch.Tensor: Scalar loss value
        
        Note:
            If a pixel in the GT is invalid (e.g., occluded, far away), it should be
            marked as 0 in the mask and will be excluded from loss computation.
        """
        # Default mask: all valid pixels
        if mask is None:
            mask = torch.ones_like(gt_depth, dtype=torch.bool)
        else:
            mask = mask.bool()
        
        # L1 loss on valid pixels only
        l1_loss = F.l1_loss(
            pred_depth[mask],
            gt_depth[mask],
            reduction='mean'
        )
        
        # SSIM loss
        # Note: SSIM is typically computed on valid regions for better perceptual quality
        ssim_loss = 1.0 - self.compute_ssim(pred_depth, gt_depth)
        
        # Combined loss
        total_loss = self.weight_l1 * l1_loss + self.weight_ssim * ssim_loss
        
        return total_loss


class DepthVariationLoss(nn.Module):
    """
    Regularization loss to encourage locally smooth depth predictions.
    
    Encourages the predicted depth map to be smooth except at object boundaries,
    which improves visual quality and reduces artifacts.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_depth: torch.Tensor) -> torch.Tensor:
        """
        Compute depth smoothness regularization loss.
        
        Args:
            pred_depth (torch.Tensor): Predicted depth maps [Batch, 1, Height, Width]
        
        Returns:
            torch.Tensor: Scalar regularization loss
        """
        # Compute gradients (approximation of smoothness)
        # Gradient in x direction
        grad_x = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
        # Gradient in y direction
        grad_y = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
        
        # Mean gradient magnitude
        smoothness_loss = (grad_x.mean() + grad_y.mean()) / 2
        
        return smoothness_loss
