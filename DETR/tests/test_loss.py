import torch
import unittest

from loss.HungarianMatching import HungarianMatching
from loss.ClassificationLoss import ClassificationLoss
from loss.RegressionLoss import RegressionLoss
from loss.loss import Loss
from loss.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class TestDETRLoss(unittest.TestCase):
    def setUp(self):
        self.num_classes = 5
        self.batch_size = 1
        self.num_queries = 10
        
        # Mock outputs
        self.outputs = {
            'pred_logits': torch.randn(self.batch_size, self.num_queries, self.num_classes + 1),
            'pred_boxes': torch.rand(self.batch_size, self.num_queries, 4)
        }
        
        # Mock targets
        self.targets = [{
            'labels': torch.tensor([1, 2], dtype=torch.long),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])
        }]

    def test_box_ops(self):
        # Test cxcywh to xyxy
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]]) # center at 0.5,0.5 with size 0.2
        expected = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
        converted = box_cxcywh_to_xyxy(boxes)
        self.assertTrue(torch.allclose(converted, expected))
        
        # Test GIoU (Perfect overlap should result in 1.0)
        giou = generalized_box_iou(expected, expected)
        self.assertAlmostEqual(giou.item(), 1.0, places=5)

    def test_hungarian_matcher(self):
        matcher = HungarianMatching(cost_class=1, cost_bbox=1, cost_giou=1)
        indices = matcher(self.outputs, self.targets)
        
        self.assertEqual(len(indices), self.batch_size)
        src_idx, tgt_idx = indices[0]
        # Should match exactly the number of ground truth objects
        self.assertEqual(len(src_idx), 2)
        self.assertEqual(len(tgt_idx), 2)

    def test_classification_loss(self):
        matcher = HungarianMatching()
        indices = matcher(self.outputs, self.targets)
        
        criterion = ClassificationLoss(num_classes=self.num_classes)
        loss_dict = criterion(self.outputs, self.targets, indices, num_boxes=2)
        
        self.assertIn('loss_ce', loss_dict)
        self.assertGreater(loss_dict['loss_ce'].item(), 0)

    def test_regression_loss(self):
        matcher = HungarianMatching()
        indices = matcher(self.outputs, self.targets)
        
        criterion = RegressionLoss()
        loss_dict = criterion(self.outputs, self.targets, indices, num_boxes=2)
        
        self.assertIn('loss_bbox', loss_dict)
        self.assertIn('loss_giou', loss_dict)
        self.assertGreater(loss_dict['loss_bbox'].item(), 0)

    def test_full_criterion_wrapper(self):
        matcher = HungarianMatching()
        weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        criterion = Loss(
            num_classes=self.num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict
        )
        
        losses = criterion(self.outputs, self.targets)
        
        self.assertEqual(len(losses), 3)
        for k in weight_dict.keys():
            self.assertIn(k, losses)
            self.assertFalse(torch.isnan(losses[k]))

if __name__ == '__main__':
    unittest.main()
