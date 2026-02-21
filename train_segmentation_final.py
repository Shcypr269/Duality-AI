import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
import os
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
import platform

# augmentation 
class DesertAugmentation:
    def __call__(self, img):
        # color jitter 
        if random.random() > 0.5:
            img = transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                        saturation=0.3, hue=0.1)(img)
        # gamma
        if random.random() > 0.5:
            gamma = random.uniform(0.7, 1.5)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = np.power(img_array, gamma)
            img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        
        if random.random() > 0.6:
            img_array = np.array(img).astype(np.float32)
            h, w = img_array.shape[:2]
            intensity = random.uniform(0.3, 0.7)
            direction = random.choice(['top', 'bottom', 'left', 'right'])
            
            if direction in ['top', 'bottom']:
                gradient = np.linspace(1, intensity, h)[:, None]
                if direction == 'bottom':
                    gradient = gradient[::-1]
            else:
                gradient = np.linspace(1, intensity, w)[None, :]
                if direction == 'right':
                    gradient = gradient[:, ::-1]
            
            img_array = img_array * gradient[:, :, None]
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        
        return img



value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# dataset
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.desert_aug = DesertAugmentation()
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))

        if self.augment:
            # augmentations
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
        
            image = self.desert_aug(image)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask

# model
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice_loss(pred, target) + self.ce_weight * self.ce_loss(pred, target)

def calculate_class_weights(data_loader, device, num_classes=10):
    print("Calculating class weights...")
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in tqdm(data_loader, desc="Computing weights", leave=False):
        labels = labels.squeeze(1).long()
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
    
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights.to(device)

# metric
def compute_iou_per_class(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou_per_class.append((intersection / union).cpu().numpy() if union > 0 else float('nan'))
    
    return np.nanmean(iou_per_class), iou_per_class

def evaluate_metrics(model, backbone, data_loader, device, num_classes=10):
    model.eval()
    iou_scores, all_class_ious = [], []
    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            
            labels = labels.squeeze(dim=1).long()
            iou, class_iou = compute_iou_per_class(outputs, labels, num_classes)
            iou_scores.append(iou)
            all_class_ious.append(class_iou)
    
    model.train()
    return np.mean(iou_scores), np.nanmean(all_class_ious, axis=0)

# plottinmg
def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # metric
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('train_loss', 'val_loss', 'Loss'), ('train_iou', 'val_iou', 'IoU'),
               ('learning_rate', None, 'Learning Rate'), ('val_class_iou', None, 'Per-Class IoU')]
    
    for ax, (train_key, val_key, title) in zip(axes.flat, metrics):
        if train_key == 'val_class_iou':
            class_iou_over_time = np.array(history[train_key]).T
            for i, class_iou in enumerate(class_iou_over_time):
                if not np.all(np.isnan(class_iou)):
                    ax.plot(class_iou, label=class_names[i], alpha=0.7)
            ax.legend(ncol=2, fontsize=8)
        else:
            ax.plot(history[train_key], label='train' if val_key else '', linewidth=2)
            if val_key:
                ax.plot(history[val_key], label='val', linewidth=2)
                ax.legend()
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # text
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Best Val IoU: {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"Final Val IoU: {history['val_iou'][-1]:.4f}\n")
        best_epoch_idx = np.argmax(history['val_iou'])
        f.write(f"\nPer-Class IoU at Best Epoch:\n")
        for name, iou in zip(class_names, history['val_class_iou'][best_epoch_idx]):
            f.write(f"  {name:<20}: {iou:.4f if not np.isnan(iou) else 'N/A'}\n")

# main
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # config
    batch_size, lr, n_epochs, warmup_epochs, patience = 2, 3e-4, 25, 2, 5
    w, h = int(((960 / 2) // 14) * 14), int(((540 / 2) // 14) * 14)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_compact')
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
    
    # transforms
    transform = transforms.Compose([
        transforms.Resize((h, w)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor()])
    
    # dataset
    num_workers = 0 if platform.system() == 'Windows' else 2
    trainset = MaskDataset(data_dir, transform, mask_transform, augment=True)
    valset = MaskDataset(val_dir, transform, mask_transform, augment=False)
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train: {len(trainset)}, Val: {len(valset)}")
    
    # class weights
    class_weights = calculate_class_weights(train_loader, device, n_classes)
    
    # backbone
    print("DINOv2...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    
    # embedding dim
    with torch.no_grad():
        sample_output = backbone.forward_features(next(iter(train_loader))[0].to(device))["x_norm_patchtokens"]
    n_embedding = sample_output.shape[2]
    
    # Model
    classifier = SegmentationHeadConvNeXt(n_embedding, n_classes, w // 14, h // 14).to(device)
    
    # Loss, optimizer, scheduler, scaler
    loss_fct = CombinedLoss(class_weights, dice_weight=0.5, ce_weight=0.5)
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
        lambda e: (e + 1) / warmup_epochs if e < warmup_epochs else 0.5 * (1 + np.cos(np.pi * (e - warmup_epochs) / (n_epochs - warmup_epochs))))
    scaler = GradScaler()
    
    # Training
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 
               'learning_rate': [], 'val_class_iou': []}
    best_val_iou, epochs_no_improve, best_state = 0.0, 0, None
    
    print(f"\nTraining: {n_epochs} epochs, lr={lr}, warmup={warmup_epochs}, patience={patience}")
    print("Improvements: Augmentation + Dice+WCE + AdamW + Scheduler + AMP + Early Stop\n")
    
    for epoch in tqdm(range(n_epochs), desc="Training"):
        # Train
        classifier.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                with torch.no_grad():
                    output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels.squeeze(1).long())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        
        # Validate
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = classifier(output.to(device))
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    loss = loss_fct(outputs, labels.squeeze(1).long())
                val_losses.append(loss.item())
        
        # Metrics
        train_iou, _ = evaluate_metrics(classifier, backbone, train_loader, device, n_classes)
        val_iou, val_class_iou = evaluate_metrics(classifier, backbone, val_loader, device, n_classes)
        
        # History
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['val_class_iou'].append(val_class_iou)
        
        scheduler.step()
        
        # Early stopping
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            best_state = classifier.state_dict().copy()
            torch.save(best_state, os.path.join(script_dir, "segmentation_head_best.pth"))
        else:
            epochs_no_improve += 1
        
        # Log every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}: Val IoU={val_iou:.4f}, Best={best_val_iou:.4f}, Patience={epochs_no_improve}/{patience}")
            print("Per-class IoU:", ", ".join([f"{name[:3]}:{iou:.2f}" for name, iou in zip(class_names, val_class_iou) if not np.isnan(iou)]))
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save
    if best_state:
        classifier.load_state_dict(best_state)
    torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head.pth"))
    save_plots(history, output_dir)
    
    print(f"\n{'='*60}")
    print(f"FINAL: Best IoU={best_val_iou:.4f}, Final IoU={history['val_iou'][-1]:.4f}")
    print(f"Models saved: segmentation_head.pth, segmentation_head_best.pth")
    print(f"Plots saved: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":

    main()
