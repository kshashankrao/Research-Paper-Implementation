import torch
import torch.nn as nn

class DETRPredictionHeads(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80):
        super().__init__()
        
        # Head to predect the class of each object query.
        # Maps tensor of size [Batch, 100, 256] to [Batch, 100, num_classes + 1]
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        
        # Head to predict the bounding box coordinates for each object query. The output is 4 values: (cx, cy, w, h)
        # A 3-layer MLP mapping 256 -> 256 -> 256 -> 4 (cx, cy, w, h)
        self.bbox_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, 4))

    def forward(self, decoder_output):
        # Input: [Batch, 100, 256]
        # Predict classes: [Batch, 100, 81]
        pred_logits = self.class_head(decoder_output)
        
        # Predict bounding boxes and apply sigmoid to constrain coordinates to [0, 1]
        # Shape: [Batch, 100, 4]
        pred_boxes = self.bbox_head(decoder_output).sigmoid()
        
        return pred_logits, pred_boxes