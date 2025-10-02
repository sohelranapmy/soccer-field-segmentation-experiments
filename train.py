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
from torch.utils.tensorboard import SummaryWriter
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
        self.image_files = self.image_files  # Limit to first 100 images for debugging
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
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)  # Convert to (1, H, W) if single channel
        # target_weight = np.ones((mask.shape[0], 1), dtype=np.float32)
        # for i,keypoint in enumerate(mask):
        #     sum_kp = np.sum(keypoint)
        #     if sum_kp!=0:
        #         target_weight[i] = 1
        #     else:
        #         target_weight[i] = 0
            
        mask = np.transpose(mask, (1, 2, 0))
        resized_image=cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
        
        if self.transform:
            augmented = self.transform(image=resized_image)
            image = augmented['image']
        
        # Pad image to be divisible by 32 (required for HRNet)
        h, w = image.shape[1], image.shape[2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Calculate expected output size after padding (HRNet highest resolution is 1/2 scale)
        padded_h = h + pad_h
        padded_w = w + pad_w
        output_h = padded_h // 2
        output_w = padded_w // 2
        
        # Resize mask to match expected output size
        resized_mask = cv2.resize(mask, (output_w, output_h), interpolation=cv2.INTER_AREA)
        if len(resized_mask.shape) == 2:
            resized_mask = np.expand_dims(resized_mask, axis=2) 
        resized_mask = torch.from_numpy(resized_mask)
        mask = resized_mask.permute(2, 0, 1)
        
        # target_weight = torch.from_numpy(target_weight)
        target_weight = None

            # mask = augmented['mask']
        # Transpose back to (C, H, W)
        # mask = mask.permute(2, 0, 1)
        # import pdb; pdb.set_trace()
        return image.float(), mask.float()#,target_weight

# ----------------------------
# HRNet Segmentation Model
# ----------------------------
class HRNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('hrnet_w48', pretrained=True, features_only=True)
        # Get the number of channels from all feature maps and sum them
        feature_channels = sum([info['num_chs'] for info in self.backbone.feature_info])
        self.head = nn.Sequential(
            nn.Conv2d(feature_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, input):
        # Get all feature maps from HRNet
        features = self.backbone(input)
        # Resize all features to match the highest resolution (first feature map)
        target_size = features[0].shape[2:]
        upsampled_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        # Concatenate all features
        x = torch.cat(upsampled_features, dim=1)
        x = self.head(x)
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
            axes[0].imshow(gt, cmap='hot', vmin=0, vmax=1)
            axes[0].set_title(f"GT - Class {c}" if not class_names else f"GT - {class_names[c]}")
            axes[0].axis("off")

            axes[1].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f"Pred - Class {c}" if not class_names else f"Pred - {class_names[c]}")
            axes[1].axis("off")

            # axes[0, c].imshow(gt, cmap='hot', vmin=0, vmax=1)
            # axes[0, c].set_title(f"GT - Class {c}" if not class_names else f"GT - {class_names[c]}")
            # axes[0, c].axis("off")

            # axes[1, c].imshow(pred, cmap='hot', vmin=0, vmax=1)
            # axes[1, c].set_title(f"Pred - Class {c}" if not class_names else f"Pred - {class_names[c]}")
            # axes[1, c].axis("off")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i+1}.png")
        plt.savefig(save_path)
        plt.close(fig)  # Don't keep figures open in memory


# ----------------------------
# Training & Validation Loops
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, writer=None, epoch=None):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    # for images, masks,target_weight in tqdm(dataloader, desc="Training", leave=False):
    for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Training", leave=False)):

        images = images.to(device)
        masks = masks.to(device)
        # target_weight = target_weight.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # outputs = torch.sigmoid(outputs)
        # loss = criterion(outputs, masks,target_weight)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Log batch loss to TensorBoard
        if writer is not None and epoch is not None:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

    avg_loss = total_loss / len(dataloader)
    
    # Log epoch loss to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, save_dir='./visualisation/', plot_samples=True, writer=None):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_pixels = 0

    with torch.no_grad():
        # for batch_idx, (images, masks,target_weight) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):

            images = images.to(device)
            masks = masks.to(device)
            # target_weight = target_weight.to(device)

            outputs = model(images)
            # outputs = torch.sigmoid(outputs)
            # loss = criterion(outputs, masks,target_weight)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate pixel-wise accuracy (for heatmap regression, use threshold-based accuracy)
            preds_binary = (outputs > 0.5).float()
            targets_binary = (masks > 0.5).float()
            correct_predictions += (preds_binary == targets_binary).sum().item()
            total_pixels += masks.numel()
            
            if epoch%20==0:
                if batch_idx == 0 and plot_samples:
                    preds = outputs
                    # preds = torch.sigmoid(outputs)
                    print("Sigmoid output stats:", preds.min().item(), preds.max().item(), preds.mean().item())
                    plot_heatmaps(images, masks, preds, max_samples=10, save_dir=save_dir, epoch=epoch)
                    
                    # Log sample images to TensorBoard
                    if writer is not None:
                        # Log original images
                        writer.add_images('Validation/Images', images[:4], epoch)
                        # Log ground truth heatmaps
                        writer.add_images('Validation/GT_Heatmaps', masks[:4], epoch)
                        # Log predicted heatmaps
                        writer.add_images('Validation/Pred_Heatmaps', preds[:4], epoch)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_pixels
    
    # Log validation metrics to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/Validation', avg_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

    return avg_loss

# ----------------------------
# Main Training Script
# ----------------------------
def main():
    # === Config ===
    num_classes = 58
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === TensorBoard Setup ===
    log_dir = 'runs/brighton_soccer_field_segmentation'
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print(f"🚀 Start TensorBoard with: tensorboard --logdir={log_dir}")

    # === Transforms ===
    transform = A.Compose([
        # A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])

    # === Data paths ===
    train_image_dir = '/home/training-machine/Documents/brighton-project/soccer-field-segmentation-experiments/brighton_data/train_rescaled_images'
    train_mask_dir = '/home/training-machine/Documents/brighton-project/soccer-field-segmentation-experiments/brighton_data/train_rescaled_images_heatmaps'
    val_image_dir = '/home/training-machine/Documents/brighton-project/soccer-field-segmentation-experiments/brighton_data/valid_rescaled_images'
    val_mask_dir = '/home/training-machine/Documents/brighton-project/soccer-field-segmentation-experiments/brighton_data/valid_rescaled_images_heatmaps'

    # === Dataset and DataLoader ===
    train_dataset = SoftHeatmapSegmentationDataset(train_image_dir, transform=transform)
    val_dataset = SoftHeatmapSegmentationDataset(val_image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    num_classes = 1
    # === Model, Loss, Optimizer ===
    model = HRNetSegmentation(num_classes=num_classes).to(device)
    # model = TinySegmentationModel(num_classes=num_classes).to(device)
    criterion = nn.MSELoss(reduction='sum')#JointsMSELoss(use_target_weight=True)#nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Log hyperparameters to TensorBoard
    writer.add_hparams({
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_classes': num_classes,
        'weight_decay': 1e-4
    }, {})

    best_val_loss = float('inf')

    # === Training Loop ===
    for epoch in range(num_epochs):
        print(f"\n📘 Epoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, writer, epoch)
        
        val_loss = validate(model, val_loader, criterion, device, epoch, save_dir='./visalisation', plot_samples=True, writer=writer)
        scheduler.step(val_loss)
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"📉 Train Loss: {train_loss:.8f} | 🧪 Val Loss: {val_loss:.8f} | 📈 LR: {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "hrnet_best_model.pth")
            print("✅ Best model saved.")
            # Log best model metrics
            writer.add_scalar('Best_Validation_Loss', best_val_loss, epoch)

    # Close TensorBoard writer
    writer.close()
    print("\n🎉 Training complete.")
    print(f"📊 View training logs with: tensorboard --logdir={log_dir}")

if __name__ == '__main__':
    main()
