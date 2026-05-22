import torch
from torch import nn
from .classificationloss import ClassificationLoss
from .regressionloss import RegressionLoss

class Loss(nn.Module):
    """ 
    This class computes the loss for DETR.
    The process happens in two steps:
        1) compute hungarian matching between ground truth boxes and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction (plus un-matched predictions as background)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        self.classification_loss = ClassificationLoss(num_classes, eos_coef)
        self.regression_loss = RegressionLoss()

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute the losses
        losses = {}
        losses.update(self.classification_loss(outputs, targets, indices, num_boxes))
        losses.update(self.regression_loss(outputs, targets, indices, num_boxes))
        
        # Apply weights from weight_dict
        weighted_losses = {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        return weighted_losses