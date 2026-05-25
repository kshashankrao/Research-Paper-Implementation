"""Loss module for MonoDETR."""

from .Hungarian3dBboxMatching import Hungarian3dBboxMatching
from .DepthLoss import DepthLoss, DepthVariationLoss
from .ClassificationLoss import ClassificationLoss, WeightedClassificationLoss
from .RegressionLoss import Box2dRegressionLoss, RegressionLoss, CombinedRegressionLoss
from .Loss import Loss
from .box_ops import (
    generalized_box3d_iou,
    generalized_box_iou_2d,
    box3d_to_2d,
    get_src_permutation_idx
)

__all__ = [
    'Hungarian3dBboxMatching',
    'DepthLoss',
    'DepthVariationLoss',
    'ClassificationLoss',
    'WeightedClassificationLoss',
    'Box2dRegressionLoss',
    'RegressionLoss',
    'CombinedRegressionLoss',
    'Loss',
    'generalized_box3d_iou',
    'generalized_box_iou_2d',
    'box3d_to_2d',
    'get_src_permutation_idx'
]
