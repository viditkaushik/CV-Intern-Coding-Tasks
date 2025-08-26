# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from roboflow import Roboflow
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from model import QueryDETR

class ConstructionWorkersDataset(Dataset):
    """Custom PyTorch Dataset for construction workers object detection data."""
    def __init__(self, data_dir="train", transform=None):
        import os
        
        self.transform = transform
        self.data_dir = data_dir
        annotation_file = os.path.join(self.data_dir, "_annotations.coco.json")

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.class_map = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
        self.idx_to_class = {i: cat['name'] for i, cat in enumerate(coco_data['categories'])}
        self.num_classes = len(coco_data['categories'])

        img_id_map = {img['id']: os.path.join(self.data_dir, img['file_name']) for img in coco_data['images']}
        self.annotations = {img_id: [] for img_id in img_id_map}
        for ann in coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        self.image_files = list(img_id_map.values())
        self.img_ids = list(img_id_map.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes, labels = [], []
        for ann in self.annotations[self.img_ids[idx]]:
            x1, y1, bw, bh = ann['bbox']
            boxes.append([(x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h])
            labels.append(self.class_map[ann['category_id']])

        target = {'boxes': torch.as_tensor(boxes, dtype=torch.float32), 
                  'labels': torch.as_tensor(labels, dtype=torch.int64)}

        if self.transform:
            image = self.transform(image)
        
        return image, target

def collate_fn(batch):
    """Custom collate function to handle varying numbers of objects."""
    return tuple(zip(*batch))

def plot_losses(train_losses, val_losses, epochs):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True); plt.savefig('loss_curves.png'); plt.show()

def box_cxcywh_to_xyxy(x):
    """Converts bbox from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)

def visualize_predictions(image, predictions, targets, idx_to_class, confidence_threshold=0.7):
    """Visualizes model predictions on a single image."""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean # Un-normalize
    img_np = np.clip(img_np, 0, 1)
    img_h, img_w, _ = img_np.shape

    fig, ax = plt.subplots(1, figsize=(12, 9)); ax.imshow(img_np)

    # Plot Ground Truth
    for box, label in zip(targets['boxes'], targets['labels']):
        box_unnorm = box * torch.tensor([img_w, img_h, img_w, img_h])
        x1, y1, x2, y2 = box_cxcywh_to_xyxy(box_unnorm.unsqueeze(0)).squeeze(0)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"GT: {idx_to_class[label.item()]}", color='white', backgroundcolor='green', fontsize=8)

    # Plot Predictions
    probas = predictions['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > confidence_threshold
    
    for box, label_idx, score in zip(predictions['pred_boxes'][0, keep], probas[keep].argmax(-1), probas[keep].max(-1).values):
        box_unnorm = box * torch.tensor([img_w, img_h, img_w, img_h])
        x1, y1, x2, y2 = box_cxcywh_to_xyxy(box_unnorm.unsqueeze(0)).squeeze(0)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, f"Pred: {idx_to_class[label_idx.item()]} ({score:.2f})", color='white', backgroundcolor='red', fontsize=8)
        
    plt.axis('off'); plt.savefig('sample_detection.png'); plt.show()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, no_object_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        weight = torch.ones(num_classes + 1)
        weight[num_classes] = no_object_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.l1_loss = nn.L1Loss()

    @torch.no_grad()
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        pred_logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']
        
        indices = []
        for i in range(pred_logits.shape[0]):
            tgt_labels, tgt_boxes = targets[i]['labels'], targets[i]['boxes']
            if len(tgt_labels) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            cost_class = -pred_logits[i].softmax(-1).detach()[:, tgt_labels]
            cost_bbox = torch.cdist(pred_boxes[i].detach(), tgt_boxes, p=1)
            row_ind, col_ind = linear_sum_assignment((cost_bbox + cost_class).cpu())
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))

        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[batch_idx, src_idx] = target_classes_o
        loss_ce = self.ce_loss(pred_logits.transpose(1, 2), target_classes)

        matched_pred_boxes = pred_boxes[batch_idx, src_idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = self.l1_loss(matched_pred_boxes, target_boxes) if matched_pred_boxes.numel() > 0 else 0.0
        
        return loss_ce + 5 * loss_bbox


if __name__ == "__main__":

    NUM_QUERIES = 20
    EPOCHS = 10
    LR = 1e-4
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    transforms = T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Resize((480, 480), antialias=True),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Setting up dataset...")
    train_dataset = ConstructionWorkersDataset("train", transforms)
    val_dataset = ConstructionWorkersDataset("test", transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    NUM_CLASSES = train_dataset.num_classes
    print(f"Dataset has {NUM_CLASSES} classes.")

    model = QueryDETR(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(DEVICE)
    criterion = DetectionLoss(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


    train_losses, val_losses = [], []
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            images = torch.stack([img.to(DEVICE) for img in images])
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images = torch.stack([img.to(DEVICE) for img in images])
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot training/validation loss curves
    plot_losses(train_losses, val_losses, EPOCHS)
    
    # Show sample detections on validation images
    print("Showing sample detections...")
    model.eval()
    with torch.no_grad():
        for i in range(3):  # Show 3 sample detections
            img, tgt = val_dataset[np.random.randint(len(val_dataset))]
            outputs = model(img.unsqueeze(0).to(DEVICE))
            visualize_predictions(img, outputs, tgt, train_dataset.idx_to_class)