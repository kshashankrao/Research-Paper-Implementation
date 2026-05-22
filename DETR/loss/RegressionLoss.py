import torch
import torch.nn.functional as F
from torch import nn
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, get_src_permutation_idx

class RegressionLoss(nn.Module):
    def forward(self, outputs, targets, indices, num_boxes):
        """Regression loss: L1 and GIoU"""
        idx = get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 1. L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses