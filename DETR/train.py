import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import AdamW
import torch.nn as nn

from model.DETR import DETR
from loss.matcher import HungarianMatcher
from loss.criterion import SetCriterion

class MNISTDetection(datasets.MNIST):
    """
    A wrapper to convert MNIST classification to a detection-like format.
    Targets are returned as a dict with 'labels' and 'boxes'.
    """
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        
        # DETR expects boxes in [cx, cy, w, h] format, normalized to [0, 1]
        # For MNIST, we assume the digit occupies the center and most of the frame.
        target = {
            'labels': torch.tensor([label], dtype=torch.long),
            'boxes': torch.tensor([[0.5, 0.5, 0.9, 0.9]], dtype=torch.float32)
        }
        return img, target

def collate_fn(batch):
    """
    Custom collate to handle the list of dictionaries in DETR targets.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def train():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10 # 0-9 digits
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 10

    # 2. Dataset and Dataloader
    # We resize to 224x224 so that ResNet50's 1/32 downsampling results in a 7x7 feature map
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Grayscale to RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MNISTDetection(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 3. Model, Matcher, and Criterion
    model = DETR(num_classes=num_classes, hidden_dim=256, num_queries=20).to(device)
    
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    
    weight_dict = {
        'loss_ce': 1,
        'loss_bbox': 5,
        'loss_giou': 2
    }
    
    criterion = SetCriterion(num_classes=num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(device)

    # 4. Optimizer
    # Standard DETR uses AdamW with a lower learning rate for the backbone, 
    # but for this MNIST example, a single LR is fine.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. Training Loop
    model.train()
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            outputs = model(images)
            
            # Compute losses
            loss_dict = criterion(outputs, targets)
            
            # Sum all weighted losses
            losses = sum(loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Gradient clipping is often used in Transformers to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()

            total_loss += losses.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {losses.item():.4f} (CE: {loss_dict['loss_ce']:.4f}, "
                      f"BBox: {loss_dict['loss_bbox']:.4f}, GIoU: {loss_dict['loss_giou']:.4f})")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'detr_mnist_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    train()