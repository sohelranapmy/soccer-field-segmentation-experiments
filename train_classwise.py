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
class ImprovedJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, absent_penalty=0.1, combined_weight=1.0):
        super(ImprovedJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.absent_penalty = absent_penalty
        self.combined_weight = combined_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = output[:, idx, :, :].reshape(batch_size, -1)  # (B, H*W)
            heatmap_gt = target[:, idx, :, :].reshape(batch_size, -1)    # (B, H*W)
            weight = target_weight[:, idx, 0]  # (B,) - extract scalar weight for this class
            
            # Special handling for combined keypoints channel (last channel)
            is_combined_channel = (idx == num_joints - 1)
            
            if self.use_target_weight and not is_combined_channel:
                # Regular keypoint classes: use target weight masking
                present_mask = weight > 0.5  # (B,)
                absent_mask = weight <= 0.5   # (B,)
                
                # Apply mask to select samples with present keypoints
                if present_mask.any():
                    present_pred = heatmap_pred[present_mask]  # (N_present, H*W)
                    present_gt = heatmap_gt[present_mask]      # (N_present, H*W)
                    if present_pred.numel() > 0:
                        loss += self.criterion(present_pred, present_gt)
                
                # For absent classes, penalize high predictions
                if absent_mask.any():
                    absent_pred = heatmap_pred[absent_mask]  # (N_absent, H*W)
                    if absent_pred.numel() > 0:
                        zero_target = torch.zeros_like(absent_pred)
                        absent_loss = self.criterion(absent_pred, zero_target)
                        loss += self.absent_penalty * absent_loss
            
            elif is_combined_channel:
                # Combined keypoints channel: apply same logic but with higher weight
                present_mask = weight > 0.5  # (B,)
                absent_mask = weight <= 0.5   # (B,)
                
                if present_mask.any():
                    present_pred = heatmap_pred[present_mask]  # (N_present, H*W)
                    present_gt = heatmap_gt[present_mask]      # (N_present, H*W)
                    if present_pred.numel() > 0:
                        loss += self.combined_weight * self.criterion(present_pred, present_gt)
                
                if absent_mask.any():
                    absent_pred = heatmap_pred[absent_mask]  # (N_absent, H*W)
                    if absent_pred.numel() > 0:
                        zero_target = torch.zeros_like(absent_pred)
                        absent_loss = self.criterion(absent_pred, zero_target)
                        loss += self.combined_weight * self.absent_penalty * absent_loss
            
            else:
                # No target weight: standard MSE loss
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

# Keep original loss for backward compatibility
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
        self.image_files = self.image_files  # Limit to first 8 images for testing
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
        
        # Load classwise heatmaps (33 classes: 1 background + 32 keypoints)
        mask = mask_npz['classwise_heatmaps'].astype(np.float32)  # shape: (C, H, W)
        
        # Get keypoint classes present in this image (shifted by +1 due to background class)
        keypoint_classes = mask_npz['keypoint_classes'] if 'keypoint_classes' in mask_npz else []
        
        # Create combined keypoints heatmap (sum of all keypoint classes, excluding background)
        combined_keypoints = np.sum(mask[1:], axis=0, keepdims=True)  # Sum keypoint classes (exclude background at index 0)
        combined_keypoints = np.clip(combined_keypoints, 0, 1)  # Ensure values stay in [0, 1] range
        
        # Add combined keypoints as additional channel
        mask = np.concatenate([mask, combined_keypoints], axis=0)  # shape: (C+1, H, W)
        
        # Create target weight based on which keypoint classes are present
        target_weight = np.zeros((mask.shape[0], 1), dtype=np.float32)
        
        # Background class (index 0) always has weight 0 (not used for loss)
        target_weight[0] = 0
        
        # Set weight to 1 for keypoint classes that are present in the image
        for keypoint_id in keypoint_classes:
            class_id = keypoint_id + 1  # Shift by 1 due to background class
            if 0 < class_id < mask.shape[0] - 1:  # Ensure valid class index (exclude combined channel)
                target_weight[class_id] = 1
        
        # Combined keypoints channel (last index) has weight 1 if any keypoints are present
        target_weight[-1] = 1 if len(keypoint_classes) > 0 else 0
        
        # Convert to (H, W, C) for CV2 resize
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
        # CV2 resize can only handle up to 4 channels, so resize each channel separately
        num_channels = mask.shape[2]
        resized_mask = np.zeros((output_h, output_w, num_channels), dtype=np.float32)
        
        for c in range(num_channels):
            resized_mask[:, :, c] = cv2.resize(mask[:, :, c], (output_w, output_h), interpolation=cv2.INTER_AREA)
        
        resized_mask = torch.from_numpy(resized_mask)
        mask = resized_mask.permute(2, 0, 1)  # Convert back to (C, H, W)
        
        target_weight = torch.from_numpy(target_weight)

        # Return image, mask, and target_weight for JointsMSELoss
        return image.float(), mask.float(), target_weight.float()

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
    Plots and saves GT and predicted heatmaps for each class on separate axes.
    Assumes shape (B, C, H, W)
    """
    os.makedirs(save_dir, exist_ok=True)

    batch_size, num_classes, H, W = gts.shape
    num_to_show = min(max_samples, batch_size)

    for i in range(num_to_show):
        gt_data = gts[i].cpu().numpy()
        pred_data = preds[i].cpu().numpy()
        
        # Plot all keypoint classes (skip background class at index 0) + combined channel
        active_classes = list(range(1, num_classes-1)) + [num_classes-1]  # Individual keypoints + combined
            
        # Calculate grid dimensions
        num_active = len(active_classes)
        cols = min(4, num_active)  # Max 4 columns
        rows = (num_active + cols - 1) // cols
        
        # Create figure with subplots for each active class
        fig, axes = plt.subplots(rows, cols * 2, figsize=(5 * cols, 4 * rows))
        fig.suptitle(f"Epoch {epoch+1} - Sample {i+1} - Class-wise Heatmaps", fontsize=16)
        
        # Flatten axes for easier indexing
        if rows == 1 and cols * 2 == 1:
            axes = [axes]
        elif rows == 1 or cols * 2 == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each active class
        for idx, class_id in enumerate(active_classes):
            gt_heatmap = gt_data[class_id]
            pred_heatmap = pred_data[class_id]
            
            # Normalize heatmaps for better visualization
            gt_norm = gt_heatmap / (gt_heatmap.max() + 1e-8) if gt_heatmap.max() > 0 else gt_heatmap
            pred_norm = pred_heatmap / (pred_heatmap.max() + 1e-8) if pred_heatmap.max() > 0 else pred_heatmap
            
            # Determine if this is the combined channel
            is_combined = (class_id == num_classes - 1)
            title_suffix = "Combined" if is_combined else f"Class {class_id-1}"
            
            # Plot GT heatmap
            gt_ax_idx = idx * 2
            if gt_ax_idx < len(axes):
                im_gt = axes[gt_ax_idx].imshow(gt_norm, cmap='hot', vmin=0, vmax=1)
                axes[gt_ax_idx].set_title(f"GT {title_suffix}")
                axes[gt_ax_idx].axis("off")
                plt.colorbar(im_gt, ax=axes[gt_ax_idx], fraction=0.046, pad=0.04)
            
            # Plot Prediction heatmap
            pred_ax_idx = idx * 2 + 1
            if pred_ax_idx < len(axes):
                im_pred = axes[pred_ax_idx].imshow(pred_norm, cmap='hot', vmin=0, vmax=1)
                axes[pred_ax_idx].set_title(f"Pred {title_suffix}")
                axes[pred_ax_idx].axis("off")
                plt.colorbar(im_pred, ax=axes[pred_ax_idx], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        total_used = len(active_classes) * 2
        for j in range(total_used, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i+1}_classwise.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Don't keep figures open in memory

def create_class_color_legend(colors, num_classes, save_dir, epoch):
    """Create a color legend showing which color represents each class"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a grid showing colors for each class
    legend_items = []
    class_labels = []
    
    for c in range(1, num_classes):  # Skip background class (index 0)
        # Create a small colored rectangle
        rect = plt.Rectangle((0, 0), 1, 1, facecolor=colors[c % len(colors)][:3])
        legend_items.append(rect)
        if c == num_classes - 1:  # Combined channel
            class_labels.append(f'Combined Keypoints (Index {c})')
        else:
            class_labels.append(f'Keypoint Class {c-1} (Index {c})')  # c-1 because we shifted by +1
    
    ax.legend(legend_items, class_labels, loc='center', ncol=4, 
              title=f'Keypoint Class Colors (Epoch {epoch+1})', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    legend_path = os.path.join(save_dir, f"epoch_{epoch+1}_class_colors_legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def create_combined_heatmap_tensor(heatmaps_batch):
    """
    Create combined colored heatmap tensor for TensorBoard logging
    Args:
        heatmaps_batch: tensor of shape (B, C, H, W)
    Returns:
        tensor of shape (B, 3, H, W) with RGB colored combined heatmaps
    """
    import torch
    
    batch_size, num_classes, H, W = heatmaps_batch.shape
    device = heatmaps_batch.device
    
    # Create colormap with distinct colors for each class
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))  # Get distinct colors
    colors_tensor = torch.tensor(colors[:, :3], dtype=torch.float32, device=device)  # RGB only, same device
    
    # Initialize combined tensor (B, 3, H, W) on same device
    combined_tensor = torch.zeros((batch_size, 3, H, W), dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        for c in range(1, num_classes):  # Skip background class (index 0)
            class_heatmap = heatmaps_batch[b, c]
            
            if class_heatmap.max() > 0:  # Only process if class has data
                # Normalize class heatmap
                normalized_heatmap = class_heatmap / (class_heatmap.max() + 1e-8)
                
                # Apply class color for each RGB channel
                for ch in range(3):
                    color_intensity = normalized_heatmap * colors_tensor[c % len(colors_tensor), ch]
                    combined_tensor[b, ch] = torch.maximum(
                        combined_tensor[b, ch], 
                        color_intensity
                    )
    
    # Ensure values are in [0, 1] range for TensorBoard
    combined_tensor = torch.clamp(combined_tensor, 0, 1)
    
    return combined_tensor

# ----------------------------
# Training & Validation Loops
# ----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, writer=None, epoch=None):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, (images, masks, target_weight) in enumerate(tqdm(dataloader, desc="Training", leave=False)):

        images = images.to(device)
        masks = masks.to(device)
        target_weight = target_weight.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks, target_weight)
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
        for batch_idx, (images, masks, target_weight) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):

            images = images.to(device)
            masks = masks.to(device)
            target_weight = target_weight.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks, target_weight)
            total_loss += loss.item()
            
            # Calculate pixel-wise accuracy (for heatmap regression, use threshold-based accuracy)
            preds_binary = (outputs > 0.5).float()
            targets_binary = (masks > 0.5).float()
            correct_predictions += (preds_binary == targets_binary).sum().item()
            total_pixels += masks.numel()
            
            if epoch%10==0:
                if batch_idx == 0 and plot_samples:
                    preds = outputs
                    # preds = torch.sigmoid(outputs)
                    # print("Sigmoid output stats:", preds.min().item(), preds.max().item(), preds.mean().item())
                    plot_heatmaps(images, masks, preds, max_samples=5, save_dir=save_dir, epoch=epoch)
                    
                    # Log sample images to TensorBoard
                    # if writer is not None:
                    #     # Log original images
                    #     # writer.add_images('Validation/Images', images[:4], epoch)
                    #     # # Log ground truth heatmaps (individual channels)
                    #     # writer.add_images('Validation/GT_Heatmaps', masks[:4], epoch)
                    #     # # Log predicted heatmaps (individual channels)
                    #     # writer.add_images('Validation/Pred_Heatmaps', preds[:4], epoch)
                        
                    #     # Create and log combined colored heatmaps
                    #     gt_combined_tensor = create_combined_heatmap_tensor(masks[:4])
                    #     pred_combined_tensor = create_combined_heatmap_tensor(preds[:4])
                        
                    #     writer.add_images('Validation/GT_Combined_Colored', gt_combined_tensor, epoch)
                    #     writer.add_images('Validation/Pred_Combined_Colored', pred_combined_tensor, epoch)

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
    num_classes = 34  # 32 keypoint classes + 1 background class + 1 combined keypoints class
    batch_size = 8
    num_epochs = 500  # Very short test to verify fix
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === TensorBoard Setup ===
    log_dir = 'runs/brighton_soccer_field_segmentation_classwise_absent_panelty_0.5'
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
    
    # === Model, Loss, Optimizer ===
    model = HRNetSegmentation(num_classes=num_classes).to(device)
    # model = TinySegmentationModel(num_classes=num_classes).to(device)
    criterion = ImprovedJointsMSELoss(use_target_weight=True, absent_penalty=0.50, combined_weight=2.0)
    # criterion = JointsMSELoss(use_target_weight=True)  # Original loss for comparison
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

        # Save latest model weights every epoch
        torch.save(model.state_dict(), "hrnet_latest_model.pth")
        
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
