import torch
import torch.nn.functional as F
from torch import nn
from .box_ops import get_src_permutation_idx

class ClassificationLoss(nn.Module):
    def __init__(self, num_classes, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets, indices, num_boxes):
        """Classification loss (Cross Entropy)"""
        src_logits = outputs['pred_logits']
        idx = get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {'loss_ce': loss_ce}