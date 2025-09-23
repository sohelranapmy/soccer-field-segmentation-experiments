import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

# ----------------------------
# Dataset for Soft Heatmaps
# ----------------------------
class SoftHeatmapSegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # self.mask_dir = mask_dir
        self.transform = transform
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

        # Filter files: remove .json and keep only image files
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]
        )
        # self.image_files = self.image_files[0:1]  # Limit to first 100 images for debugging
        # self.image_files = sorted(os.listdir(image_dir))
        self.mask_dir = image_dir+"_heatmaps/"
        # import pdb; pdb.set_trace()
        # self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load soft mask
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace('.jpg', '_heatmap.npz'))
        mask_npz = np.load(mask_path)
        mask = mask_npz['arr_0'].astype(np.float32)  # shape: (C, H, W)
        target_weight = np.ones((mask.shape[0], 1), dtype=np.float32)
        for i,keypoint in enumerate(mask):
            sum_kp = np.sum(keypoint)
            if sum_kp!=0:
                target_weight[i] = 1
            else:
                target_weight[i] = 0
            
        mask = np.transpose(mask, (1, 2, 0))
        resized_mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
        resized_mask = torch.from_numpy(resized_mask)
        mask = resized_mask.permute(2, 0, 1)
        resized_image=cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mask = F.pad(mask, pad=(0, 0, 0, 18), mode='constant', value=0)
        target_weight = torch.from_numpy(target_weight)
        # Transpose mask to (H, W, C) for Albumentations
        # mask = np.transpose(mask, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=resized_image)
            image = augmented['image']
            image = F.pad(image, pad=(0, 0, 0, 18), mode='constant', value=0)

            # mask = augmented['mask']
        
        # Transpose back to (C, H, W)
        # mask = mask.permute(2, 0, 1)

        return image.float(), mask.float(),target_weight.float()

# ----------------------------
# HRNet Segmentation Model
# ----------------------------
class HRNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
        self.head = nn.Sequential(
            nn.Conv2d(self.backbone.feature_info[-1]['num_chs'], 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # import pdb; pdb.set_trace()
        features = self.backbone(input)[-1]  # Output feature map from HRNet
        
        x = self.head(features)
        x = nn.functional.interpolate(x, size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
        return x

class TinySegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Heatmap Visualization
# ----------------------------
import matplotlib.pyplot as plt
import os

def plot_heatmaps(images, gts, preds, class_names=None, max_samples=5, save_dir='visualizations', epoch=0):
    """
    Plots and saves GT and predicted heatmaps for first few samples in the batch.
    Assumes shape (B, C, H, W)
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size, num_classes, H, W = gts.shape
    num_to_show = min(max_samples, batch_size)

    for i in range(num_to_show):
        fig, axes = plt.subplots(2, num_classes, figsize=(4 * num_classes, 6))
        fig.suptitle(f"Epoch {epoch+1} - Sample {i+1}", fontsize=16)

        for c in range(num_classes):
            gt = gts[i, c].cpu().numpy()
            pred = preds[i, c].cpu().numpy()

            axes[0, c].imshow(gt, cmap='hot', vmin=0, vmax=1)
            axes[0, c].set_title(f"GT - Class {c}" if not class_names else f"GT - {class_names[c]}")
            axes[0, c].axis("off")

            axes[1, c].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[1, c].set_title(f"Pred - Class {c}" if not class_names else f"Pred - {class_names[c]}")
            axes[1, c].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)  # Don't keep figures open in memory


# ----------------------------
# Training & Validation Loops
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, masks,target_weight in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        target_weight = target_weight.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, masks,target_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, epoch,save_dir='./visualisation/',plot_samples=True):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks,target_weight) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            target_weight = target_weight.to(device)

            outputs = model(images)
            # outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks,target_weight)
            total_loss += loss.item()
            
            if epoch%20==0:
                if batch_idx == 0 and plot_samples:
                    preds = outputs
                    # preds = torch.sigmoid(outputs)
                    print("Sigmoid output stats:", preds.min().item(), preds.max().item(), preds.mean().item())
                    plot_heatmaps(images, masks, preds, max_samples=10, save_dir=save_dir, epoch=epoch)
                # plot_heatmaps(images, masks, preds, max_samples=5)

    return total_loss / len(dataloader)

# ----------------------------
# Main Training Script
# ----------------------------
def main():
    # === Config ===
    num_classes = 58
    batch_size = 16
    num_epochs = 500
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Transforms ===
    transform = A.Compose([
        # A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])

    # === Data paths ===
    train_image_dir = '/home/training-machine/Documents/brighton-project/soccer_net_data/calibration-2023/train'
    train_mask_dir = '/home/training-machine/Documents/brighton-project/soccer_net_data/calibration-2023/train_heatmaps'
    val_image_dir = '/home/training-machine/Documents/brighton-project/soccer_net_data/calibration-2023/valid'
    val_mask_dir = '/home/training-machine/Documents/brighton-project/soccer_net_data/calibration-2023/valid_heatmaps'

    # === Dataset and DataLoader ===
    train_dataset = SoftHeatmapSegmentationDataset(train_image_dir, transform=transform)
    val_dataset = SoftHeatmapSegmentationDataset(val_image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # === Model, Loss, Optimizer ===
    model = HRNetSegmentation(num_classes=num_classes).to(device)
    # model = TinySegmentationModel(num_classes=num_classes).to(device)
    criterion = JointsMSELoss(use_target_weight=True)#nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss = validate(model, val_loader, criterion, device, epoch,save_dir='./visalisation',plot_samples=True)
        scheduler.step(val_loss)
        print(f"📉 Train Loss: {train_loss:.4f} | 🧪 Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "hrnet_best_model.pth")
            print("✅ Best model saved.")

    print("\n🎉 Training complete.")

if __name__ == '__main__':
    main()
