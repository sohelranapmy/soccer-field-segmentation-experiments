#!/usr/bin/env python3
"""
Inference script for classwise keypoint detection using trained HRNet model.
Uses Option 1: Find keypoint locations from combined heatmap, then determine classes 
from individual classwise heatmaps.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time

# Import model from training script
import sys
sys.path.append('.')
from train_classwise import HRNetSegmentation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool

class KeypointInference:
    def __init__(self, model_path, device='cuda', num_classes=34):
        """
        Initialize inference pipeline
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on
            num_classes: Number of output classes (33 keypoint classes + 1 combined)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load model
        self.model = HRNetSegmentation(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Image preprocessing transform
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        
        print(f"✅ Model loaded from {model_path} on {self.device}")
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        Args:
            image: PIL Image or numpy array
        Returns:
            tensor: Preprocessed image tensor
            original_size: Original image dimensions (width, height)
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
            original_size = image.size  # (width, height)
        else:
            image_np = image
            original_size = (image_np.shape[1], image_np.shape[0])  # (width, height)
        
        # Resize image to half size (matching training)
        resized_image = cv2.resize(image_np, (original_size[0] // 2, original_size[1] // 2), 
                                 interpolation=cv2.INTER_AREA)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=resized_image)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
        
        # Pad image to be divisible by 32 (required for HRNet)
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor, original_size, (pad_h, pad_w)
    
    def detect_keypoints_from_combined(self, combined_heatmap, threshold=0.3, min_distance=10, max_keypoints=20):
        """
        Detect keypoint locations from combined heatmap using maxpool-based peak detection
        Args:
            combined_heatmap: Combined heatmap (H, W)
            threshold: Minimum confidence threshold
            min_distance: Minimum distance between peaks
            max_keypoints: Maximum number of keypoints to detect
        Returns:
            List of detected keypoint locations [(x, y, confidence), ...]
        """
        # Convert to tensor and add batch/channel dimensions for the utils function
        heatmap_tensor = torch.from_numpy(combined_heatmap).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Use the optimized maxpool-based detection
        detected = get_keypoints_from_heatmap_batch_maxpool(
            heatmap_tensor,
            scale=1,  # No scaling needed as we're working at inference resolution
            max_keypoints=max_keypoints,
            min_keypoint_pixel_distance=min_distance,
            return_scores=True
        )
        
        # Extract keypoints from the result (batch=0, channel=0)
        keypoints = []
        if detected.shape[0] > 0 and detected.shape[1] > 0:
            batch_0_channel_0 = detected[0, 0]  # Get first batch, first channel
            for detection in batch_0_channel_0:
                if len(detection) >= 3:  # x, y, confidence
                    x, y, conf = detection[0].item(), detection[1].item(), detection[2].item()
                    # Apply threshold filter
                    if conf >= threshold:
                        keypoints.append((x, y, conf))
        
        # Sort by confidence (highest first)
        keypoints = sorted(keypoints, key=lambda x: x[2], reverse=True)
        
        return keypoints
    
    def determine_keypoint_classes(self, keypoints, classwise_heatmaps, search_radius=5, 
                                  prevent_duplicates=True):
        """
        Determine class for each detected keypoint by checking individual class heatmaps
        Uses maxpool2d for more efficient and robust peak detection within search regions
        Args:
            keypoints: List of keypoint locations [(x, y, confidence), ...]
            classwise_heatmaps: Individual class heatmaps (C, H, W) - excluding combined channel
            search_radius: Radius to search around keypoint location
            prevent_duplicates: Whether to prevent multiple detections of same class
        Returns:
            Tuple of (classified_keypoints, unclassified_keypoints, used_classes)
            classified_keypoints: [(x, y, confidence, class_id, class_confidence), ...]
            unclassified_keypoints: [(x, y, confidence), ...]
            used_classes: set of class IDs that have been assigned
        """
        classified_keypoints = []
        unclassified_keypoints = []
        num_classes = classwise_heatmaps.shape[0] - 1  # Exclude background class
        used_classes = set()  # Track which classes have been assigned
        
        # Convert heatmaps to tensor for maxpool operations
        heatmaps_tensor = torch.from_numpy(classwise_heatmaps).unsqueeze(0)  # Add batch dimension
        
        # Sort keypoints by combined confidence (process highest confidence first)
        sorted_keypoints = sorted(keypoints, key=lambda x: x[2], reverse=True)
        
        for x, y, combined_conf in sorted_keypoints:
            # Ensure coordinates are integers
            x, y = int(x), int(y)
            
            # Search in a small region around the detected point
            y_min = max(0, y - search_radius)
            y_max = min(classwise_heatmaps.shape[1], y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(classwise_heatmaps.shape[2], x + search_radius + 1)
            
            # Calculate region size for maxpool
            region_h = y_max - y_min
            region_w = x_max - x_min
            
            if region_h <= 0 or region_w <= 0:
                unclassified_keypoints.append((x, y, combined_conf))
                continue
                
            best_class_id = -1
            best_class_conf = 0.0
            
            # Check each keypoint class (skip background class at index 0)
            for class_id in range(1, num_classes + 1):  # Classes 1-32
                keypoint_id = class_id - 1  # Convert to 0-31 range for keypoint IDs
                
                # Skip if this class is already assigned and duplicates are prevented
                if prevent_duplicates and keypoint_id in used_classes:
                    continue
                
                # Extract region for this class
                class_region = heatmaps_tensor[0, class_id, y_min:y_max, x_min:x_max]
                
                if class_region.numel() > 0:
                    # Simply get the maximum value in the region since each heatmap has only one class
                    max_conf = class_region.max().item()
                    
                    if max_conf > best_class_conf:
                        best_class_conf = max_conf
                        best_class_id = keypoint_id
            
            if best_class_id >= 0:
                classified_keypoints.append((x, y, combined_conf, best_class_id, best_class_conf))
                if prevent_duplicates:
                    used_classes.add(best_class_id)
            else:
                unclassified_keypoints.append((x, y, combined_conf))
        
        return classified_keypoints, unclassified_keypoints, used_classes
    
    def assign_remaining_keypoints(self, unclassified_keypoints, classwise_heatmaps, used_classes, 
                                 search_radius=5, num_classes=32):
        """
        Assign unclassified keypoints to remaining classes after initial high-confidence assignment
        Args:
            unclassified_keypoints: List of unclassified keypoints [(x, y, confidence), ...]
            classwise_heatmaps: Individual class heatmaps (C, H, W)
            used_classes: Set of class IDs already assigned
            search_radius: Radius to search around keypoint location
            num_classes: Total number of keypoint classes (32)
        Returns:
            List of newly classified keypoints [(x, y, confidence, class_id, class_confidence), ...]
        """
        remaining_keypoints = []
        available_classes = set(range(num_classes)) - used_classes
        
        if not available_classes or not unclassified_keypoints:
            return remaining_keypoints
        
        # Convert heatmaps to tensor for operations
        heatmaps_tensor = torch.from_numpy(classwise_heatmaps).unsqueeze(0)
        
        # Sort unclassified keypoints by detection confidence (highest first)
        sorted_unclassified = sorted(unclassified_keypoints, key=lambda x: x[2], reverse=True)
        
        # Track which classes get assigned in this round
        newly_used_classes = set()
        
        for x, y, combined_conf in sorted_unclassified:
            x, y = int(x), int(y)
            
            # Search region around keypoint
            y_min = max(0, y - search_radius)
            y_max = min(classwise_heatmaps.shape[1], y + search_radius + 1)
            x_min = max(0, x - search_radius)
            x_max = min(classwise_heatmaps.shape[2], x + search_radius + 1)
            
            region_h = y_max - y_min
            region_w = x_max - x_min
            
            if region_h <= 0 or region_w <= 0:
                continue
            
            best_class_id = -1
            best_class_conf = 0.0
            
            # Check only available classes
            for keypoint_id in available_classes:
                if keypoint_id in newly_used_classes:
                    continue
                    
                class_id = keypoint_id + 1  # Convert to 1-33 range for heatmap indexing
                
                # Extract region for this class
                class_region = heatmaps_tensor[0, class_id, y_min:y_max, x_min:x_max]
                
                if class_region.numel() > 0:
                    max_conf = class_region.max().item()
                    
                    if max_conf > best_class_conf:
                        best_class_conf = max_conf
                        best_class_id = keypoint_id
            
            if best_class_id >= 0:
                remaining_keypoints.append((x, y, combined_conf, best_class_id, best_class_conf))
                newly_used_classes.add(best_class_id)
        
        return remaining_keypoints
    
    def predict(self, image, detection_threshold=0.3, classification_threshold=0.2, 
               min_distance=10, search_radius=5, prevent_duplicate_classes=True, max_keypoints=20):
        """
        Run full inference pipeline on an image
        Args:
            image: Input image (PIL Image or numpy array)
            detection_threshold: Threshold for keypoint detection on combined heatmap
            classification_threshold: Minimum confidence for class assignment
            min_distance: Minimum distance between detected keypoints
            search_radius: Search radius for class determination
        Returns:
            dict: Inference results containing keypoints, heatmaps, and metadata
        """
        # Preprocess image
        image_tensor, original_size, padding = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            outputs = outputs.cpu().numpy()[0]  # Remove batch dimension
        
        # Rescale outputs to match original image coordinates
        # Network outputs at 1/4 scale, so rescale dimensions by 4x
        target_output_h = original_size[1] // 4  # Quarter of original height
        target_output_w = original_size[0] // 4  # Quarter of original width
        
        # print(f"Original size: {original_size}")
        # print(f"Target output size: {target_output_w}x{target_output_h}")
        # print(f"Actual output size: {outputs.shape[2]}x{outputs.shape[1]}")
        
        # Resize outputs to target dimensions using interpolation
        # This handles any padding/cropping issues by direct rescaling
        resized_outputs = []
        for i in range(outputs.shape[0]):
            resized_channel = cv2.resize(outputs[i], (target_output_w, target_output_h), 
                                       interpolation=cv2.INTER_LINEAR)
            resized_outputs.append(resized_channel)
        outputs = np.stack(resized_outputs, axis=0)
        
        # print(f"After rescaling: {outputs.shape[2]}x{outputs.shape[1]}")

        # Split outputs into classwise heatmaps and combined heatmap
        classwise_heatmaps = outputs[:-1]  # First 33 channels (background + 32 keypoint classes)
        combined_heatmap = outputs[-1]     # Last channel (combined keypoints)
        
        # Step 1: Detect keypoints from combined heatmap
        detected_keypoints = self.detect_keypoints_from_combined(
            combined_heatmap, threshold=detection_threshold, min_distance=min_distance,
            max_keypoints=max_keypoints
        )
        
        # Step 2: Determine classes for detected keypoints (first stage - high confidence)
        classified_keypoints, unclassified_keypoints, used_classes = self.determine_keypoint_classes(
            detected_keypoints, classwise_heatmaps, search_radius=search_radius,
            prevent_duplicates=prevent_duplicate_classes
        )
        
        # Filter by classification confidence (first stage)
        high_conf_keypoints = [
            kp for kp in classified_keypoints if kp[4] >= classification_threshold
        ]
        
        # Update used classes based on high confidence assignments
        final_used_classes = set()
        for kp in high_conf_keypoints:
            final_used_classes.add(kp[3])  # kp[3] is class_id
        
        # Step 3: Assign remaining detections to unused classes (second stage - any confidence)
        remaining_keypoints = self.assign_remaining_keypoints(
            unclassified_keypoints, classwise_heatmaps, final_used_classes,
            search_radius=search_radius, num_classes=32
        )
        
        # Combine high confidence and remaining assignments
        classified_keypoints = high_conf_keypoints + remaining_keypoints
        
        # Calculate automatic heatmap offset based on preprocessing padding
        # The padding at half-resolution gets scaled by 4x (2x from resize + 2x from network)
        pad_h, pad_w = padding
        automatic_offset = pad_h * 2  # Scale padding from half-res to full-res
        
        # Scale keypoints back to original image coordinates
        # Since we rescaled to exactly 1/4 of original size, scaling factor is exactly 4.0
        scale_x = 4.0
        scale_y = 4.0
        # print(f"Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
        # print(f"Heatmap shape: {combined_heatmap.shape}")
        # print(f"Applying automatic offset: {automatic_offset} pixels to keypoints")
        
        scaled_keypoints = []
        for x, y, det_conf, class_id, class_conf in classified_keypoints:
            scaled_x = x * scale_x
            scaled_y = y * scale_y - automatic_offset  # Apply offset correction to y-coordinate
            scaled_keypoints.append({
                'x': scaled_x,
                'y': scaled_y,
                'keypoint_id': class_id,
                'detection_confidence': det_conf,
                'classification_confidence': class_conf,
                'combined_confidence': (det_conf + class_conf) / 2
            })
        
        return {
            'keypoints': scaled_keypoints,
            'raw_keypoints': classified_keypoints,
            'classwise_heatmaps': classwise_heatmaps,
            'combined_heatmap': combined_heatmap,
            'original_size': original_size,
            'heatmap_size': combined_heatmap.shape,
            'num_detected': len(scaled_keypoints),
            'automatic_heatmap_offset': automatic_offset,
            'detection_params': {
                'detection_threshold': detection_threshold,
                'classification_threshold': classification_threshold,
                'min_distance': min_distance,
                'search_radius': search_radius
            }
        }
    
    def visualize_results(self, image, results, save_path=None, show_heatmaps=True):
        """
        Visualize inference results
        Args:
            image: Original input image
            results: Inference results from predict()
            save_path: Path to save visualization
            show_heatmaps: Whether to show heatmap overlay
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Create figure with subplots
        if show_heatmaps:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))
            axes = [axes]
        
        # Plot 1: Original image with keypoints
        axes[0].imshow(image_np)
        axes[0].set_title(f'Detected Keypoints ({len(results["keypoints"])})')
        
        # Draw detected keypoints
        for kp in results['keypoints']:
            x, y = kp['x'], kp['y']
            class_id = kp['keypoint_id']
            combined_conf = kp['combined_confidence']
            
            # Draw keypoint
            circle = patches.Circle((x, y), radius=8, color='red', fill=False, linewidth=2)
            axes[0].add_patch(circle)
            
            # Add class ID and confidence
            axes[0].text(x + 10, y - 10, f'{class_id}\n{combined_conf:.2f}', 
                        color='white', fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        axes[0].axis('off')
        
        if show_heatmaps:
            # Plot 2: Combined heatmap
            combined_heatmap = results['combined_heatmap']
            im2 = axes[1].imshow(combined_heatmap, cmap='hot', alpha=0.8)
            axes[1].set_title('Combined Heatmap')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot 3: Overlay of original image with combined heatmap
            axes[2].imshow(image_np)
            
            # Create properly aligned heatmap overlay with offset padding
            original_size = results['original_size']
            heatmap_shape = combined_heatmap.shape
            
            # Since we rescaled to exactly 1/4 of original, simply resize by 4x
            resized_heatmap = cv2.resize(combined_heatmap, original_size, interpolation=cv2.INTER_LINEAR)
            
            # Use automatic offset based on preprocessing padding
            offset_pixels = results.get('automatic_heatmap_offset', 0)
            print(f"Using heatmap offset: {offset_pixels} pixels")
            padded_heatmap = np.zeros_like(resized_heatmap)
            if offset_pixels < resized_heatmap.shape[0]:
                padded_heatmap[:-offset_pixels, :] = resized_heatmap[offset_pixels:, :]
            
            im3 = axes[2].imshow(padded_heatmap, cmap='hot', alpha=0.4)
            axes[2].set_title(f'Image + Heatmap Overlay (Offset: {offset_pixels}px up)')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def draw_keypoints_on_frame(self, frame, keypoints, draw_confidence=True, 
                               circle_radius=8, font_scale=0.6, thickness=2):
        """
        Draw keypoints directly on a video frame (in-place)
        Args:
            frame: OpenCV frame (numpy array in BGR format)
            keypoints: List of keypoint dictionaries from predict()
            draw_confidence: Whether to draw confidence scores
            circle_radius: Radius of keypoint circles
            font_scale: Font size for text
            thickness: Line thickness
        Returns:
            frame: Modified frame with keypoints drawn
        """
        frame_copy = frame.copy()
        
        for kp in keypoints:
            x, y = int(kp['x']), int(kp['y'])
            class_id = kp['keypoint_id']
            combined_conf = kp['combined_confidence']
            
            # Draw keypoint circle
            cv2.circle(frame_copy, (x, y), circle_radius, (0, 0, 255), thickness)  # Red circle
            
            if draw_confidence:
                # Draw class ID and confidence
                text = f"{class_id}: {combined_conf:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Background rectangle for text
                cv2.rectangle(frame_copy, 
                            (x + 10, y - text_size[1] - 10), 
                            (x + 10 + text_size[0], y - 5), 
                            (0, 0, 0), -1)  # Black background
                
                # Draw text
                cv2.putText(frame_copy, text, (x + 10, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return frame_copy
    
    def create_heatmap_visualization(self, heatmap, target_size, colormap=cv2.COLORMAP_JET):
        """
        Create a colored heatmap visualization
        Args:
            heatmap: 2D numpy array heatmap
            target_size: (width, height) for output size
            colormap: OpenCV colormap to use
        Returns:
            colored_heatmap: 3-channel BGR image
        """
        # Normalize heatmap to 0-255 range
        if heatmap.max() > heatmap.min():
            normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(heatmap, dtype=np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        # Resize to target size using linear interpolation for better quality
        if colored.shape[:2][::-1] != target_size:  # OpenCV uses (width, height), numpy uses (height, width)
            colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_LINEAR)
        
        return colored
    
    def create_combined_frame(self, frame, heatmap, keypoints, layout='bottom', 
                             heatmap_alpha=0.7, draw_confidence=True, results=None):
        """
        Create a combined frame with original video and heatmap visualization
        Args:
            frame: Original video frame (BGR)
            heatmap: Combined heatmap (2D array)
            keypoints: List of detected keypoints
            layout: 'bottom', 'right', or 'overlay'
            heatmap_alpha: Alpha for heatmap overlay (when layout='overlay')
            draw_confidence: Whether to draw confidence scores
        Returns:
            combined_frame: Combined visualization frame
        """
        frame_with_keypoints = self.draw_keypoints_on_frame(frame, keypoints, draw_confidence)
        
        h, w = frame.shape[:2]
        
        if layout == 'overlay':
            # Overlay heatmap directly on the frame with proper alignment and offset
            heatmap_colored = self.create_heatmap_visualization(heatmap, (w, h))
            
            # Use automatic offset based on preprocessing padding
            offset_pixels = results.get('automatic_heatmap_offset', 0) if results else 0
            padded_heatmap = np.zeros_like(heatmap_colored)
            if offset_pixels < heatmap_colored.shape[0]:
                padded_heatmap[:-offset_pixels, :] = heatmap_colored[offset_pixels:, :]
            
            combined_frame = cv2.addWeighted(frame_with_keypoints, 1 - heatmap_alpha, 
                                           padded_heatmap, heatmap_alpha, 0)
            
        elif layout == 'bottom':
            # Place heatmap at bottom (split screen vertically)
            heatmap_height = h // 2  # Make heatmap half the height
            heatmap_colored = self.create_heatmap_visualization(heatmap, (w, heatmap_height))
            
            # Add title to heatmap
            title_height = 30
            heatmap_with_title = np.zeros((heatmap_height + title_height, w, 3), dtype=np.uint8)
            heatmap_with_title[title_height:, :] = heatmap_colored
            
            # Add title text
            cv2.putText(heatmap_with_title, 'Combined Heatmap', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Concatenate vertically
            combined_frame = np.vstack([frame_with_keypoints, heatmap_with_title])
            
        elif layout == 'right':
            # Place heatmap on the right (split screen horizontally)
            heatmap_width = w // 2  # Make heatmap half the width
            heatmap_colored = self.create_heatmap_visualization(heatmap, (heatmap_width, h))
            
            # Add title to heatmap
            title_width = 150
            cv2.putText(heatmap_colored, 'Heatmap', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Concatenate horizontally
            combined_frame = np.hstack([frame_with_keypoints, heatmap_colored])
            
        else:
            # Default: just return frame with keypoints
            combined_frame = frame_with_keypoints
        
        return combined_frame
    
    def process_video(self, video_path, output_path, detection_threshold=0.3, 
                     classification_threshold=0.2, min_distance=10, search_radius=6,
                     max_frames=None, fps_limit=None, show_progress=True, prevent_duplicate_classes=True,
                     show_heatmap=False, heatmap_layout='bottom', heatmap_alpha=0.7, max_keypoints=20):
        """
        Process video and create output with keypoint overlays
        Args:
            video_path: Path to input video
            output_path: Path for output video
            detection_threshold: Threshold for keypoint detection
            classification_threshold: Threshold for classification
            min_distance: Minimum distance between keypoints
            search_radius: Search radius for class determination
            max_frames: Maximum number of frames to process (None for all)
            fps_limit: Limit output FPS (None to keep original)
            show_progress: Whether to show progress bar
            prevent_duplicate_classes: Prevent multiple detections of same class
            show_heatmap: Whether to include heatmap in output
            heatmap_layout: Layout for heatmap ('bottom', 'right', 'overlay')
            heatmap_alpha: Alpha for heatmap overlay (when layout='overlay')
        Returns:
            dict: Processing statistics
        """
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set output FPS
        output_fps = fps_limit if fps_limit is not None else original_fps
        
        # Set number of frames to process
        frames_to_process = min(max_frames, total_frames) if max_frames is not None else total_frames
        
        # Calculate output dimensions based on heatmap layout
        if show_heatmap:
            if heatmap_layout == 'bottom':
                output_width, output_height = width, height + height // 2 + 30  # Extra space for title
            elif heatmap_layout == 'right':
                output_width, output_height = width + width // 2, height
            else:  # overlay
                output_width, output_height = width, height
        else:
            output_width, output_height = width, height
        
        print(f"📹 Video info: {width}x{height} @ {original_fps:.1f} FPS, {total_frames} frames")
        print(f"📹 Processing {frames_to_process} frames @ {output_fps:.1f} FPS")
        if show_heatmap:
            print(f"🔥 Heatmap overlay: {heatmap_layout} layout, output size: {output_width}x{output_height}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
        
        # Processing statistics
        stats = {
            'total_frames': frames_to_process,
            'processed_frames': 0,
            'total_keypoints_detected': 0,
            'average_keypoints_per_frame': 0,
            'processing_time': 0,
            'fps_achieved': 0
        }
        
        start_time = time.time()
        
        # Process frames
        frame_iterator = tqdm(range(frames_to_process), desc="Processing video") if show_progress else range(frames_to_process)
        
        for frame_idx in frame_iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB for inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.predict(
                frame_rgb,
                detection_threshold=detection_threshold,
                classification_threshold=classification_threshold,
                min_distance=min_distance,
                search_radius=search_radius,
                prevent_duplicate_classes=prevent_duplicate_classes,
                max_keypoints=max_keypoints
            )
            
            # Create output frame (with or without heatmap)
            if show_heatmap:
                output_frame = self.create_combined_frame(
                    frame, results['combined_heatmap'], results['keypoints'],
                    layout=heatmap_layout, heatmap_alpha=heatmap_alpha, draw_confidence=True,
                    results=results
                )
            else:
                output_frame = self.draw_keypoints_on_frame(frame, results['keypoints'])
            
            # Write frame to output video
            out.write(output_frame)
            
            # Update statistics
            stats['processed_frames'] += 1
            stats['total_keypoints_detected'] += results['num_detected']
            
            # Update progress description if using tqdm
            if show_progress and hasattr(frame_iterator, 'set_postfix'):
                frame_iterator.set_postfix({
                    'keypoints': results['num_detected'],
                    'avg_kp': f"{stats['total_keypoints_detected'] / stats['processed_frames']:.1f}"
                })
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate final statistics
        end_time = time.time()
        stats['processing_time'] = end_time - start_time
        stats['fps_achieved'] = stats['processed_frames'] / stats['processing_time']
        stats['average_keypoints_per_frame'] = (stats['total_keypoints_detected'] / 
                                               max(stats['processed_frames'], 1))
        
        print(f"\n✅ Video processing complete!")
        print(f"   Processed: {stats['processed_frames']} frames")
        print(f"   Total keypoints: {stats['total_keypoints_detected']}")
        print(f"   Avg keypoints/frame: {stats['average_keypoints_per_frame']:.1f}")
        print(f"   Processing time: {stats['processing_time']:.1f}s")
        print(f"   Processing FPS: {stats['fps_achieved']:.1f}")
        print(f"   Output saved to: {output_path}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Classwise keypoint inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model weights')
    parser.add_argument('--input_path', required=True, help='Path to input image or video')
    parser.add_argument('--output_dir', default='inference_results', help='Output directory for results')
    parser.add_argument('--detection_threshold', type=float, default=0.3, 
                       help='Threshold for keypoint detection on combined heatmap')
    parser.add_argument('--classification_threshold', type=float, default=0.2,
                       help='Minimum confidence for class assignment')
    parser.add_argument('--min_distance', type=int, default=10,
                       help='Minimum distance between detected keypoints')
    parser.add_argument('--search_radius', type=int, default=5,
                       help='Search radius for class determination')
    parser.add_argument('--allow_duplicate_classes', action='store_true', 
                       help='Allow multiple detections of the same keypoint class')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to JSON')
    
    # Video-specific arguments
    parser.add_argument('--video_mode', action='store_true', help='Process video instead of image')
    parser.add_argument('--max_frames', type=int, help='Maximum number of frames to process')
    parser.add_argument('--fps_limit', type=float, help='Limit output video FPS')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar for video processing')
    parser.add_argument('--show_heatmap', action='store_true', help='Include heatmap visualization in video output')
    parser.add_argument('--heatmap_layout', choices=['bottom', 'right', 'overlay'], default='bottom',
                       help='Layout for heatmap visualization')
    parser.add_argument('--heatmap_alpha', type=float, default=0.7,
                       help='Alpha transparency for heatmap overlay (when layout=overlay)')
    parser.add_argument('--max_keypoints', type=int, default=20,
                       help='Maximum number of keypoints to detect per frame')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference pipeline
    inference = KeypointInference(args.model_path, device=args.device)
    
    # Check if input is video or image
    input_path = Path(args.input_path)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    is_video = args.video_mode or input_path.suffix.lower() in video_extensions
    
    if is_video:
        # Process video
        print(f"🎬 Processing video: {args.input_path}")
        
        output_video_path = os.path.join(args.output_dir, f"{input_path.stem}_keypoints{input_path.suffix}")
        
        # Process video with keypoint overlays
        stats = inference.process_video(
            video_path=args.input_path,
            output_path=output_video_path,
            detection_threshold=args.detection_threshold,
            classification_threshold=args.classification_threshold,
            min_distance=args.min_distance,
            search_radius=args.search_radius,
            max_frames=args.max_frames,
            fps_limit=args.fps_limit,
            show_progress=not args.no_progress,
            prevent_duplicate_classes=not args.allow_duplicate_classes,
            show_heatmap=args.show_heatmap,
            heatmap_layout=args.heatmap_layout,
            heatmap_alpha=args.heatmap_alpha,
            max_keypoints=args.max_keypoints
        )
        
        # Save video processing statistics
        if args.save_results:
            stats_json = {
                'input_video': args.input_path,
                'output_video': output_video_path,
                'model_path': args.model_path,
                'processing_stats': stats,
                'detection_params': {
                    'detection_threshold': args.detection_threshold,
                    'classification_threshold': args.classification_threshold,
                    'min_distance': args.min_distance,
                    'search_radius': args.search_radius
                }
            }
            
            stats_path = os.path.join(args.output_dir, f"{input_path.stem}_video_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_json, f, indent=2)
            print(f"📊 Video statistics saved to {stats_path}")
    
    else:
        # Process single image
        print(f"🖼️  Loading image: {args.input_path}")
        image = Image.open(args.input_path).convert('RGB')
        
        # Run inference
        print("🔍 Running inference...")
        results = inference.predict(
            image,
            detection_threshold=args.detection_threshold,
            classification_threshold=args.classification_threshold,
            min_distance=args.min_distance,
            search_radius=args.search_radius,
            prevent_duplicate_classes=not args.allow_duplicate_classes,
            max_keypoints=args.max_keypoints
        )
        
        # Print results summary
        print(f"\n📊 Inference Results:")
        print(f"   Detected keypoints: {results['num_detected']}")
        print(f"   Detection params: {results['detection_params']}")
        
        # Print detailed keypoint information
        if results['keypoints']:
            print(f"\n🎯 Detected Keypoints:")
            for i, kp in enumerate(results['keypoints']):
                print(f"   {i+1}. Class {kp['keypoint_id']:2d} at ({kp['x']:6.1f}, {kp['y']:6.1f}) "
                      f"- Combined conf: {kp['combined_confidence']:.3f}")
        
        # Visualize results
        image_name = input_path.stem
        viz_path = os.path.join(args.output_dir, f"{image_name}_inference.png")
        
        print(f"\n📈 Creating visualization...")
        inference.visualize_results(image, results, save_path=viz_path)
        
        # Save detailed results if requested
        if args.save_results:
            # Convert numpy arrays to lists for JSON serialization
            results_json = {
                'keypoints': results['keypoints'],
                'original_size': results['original_size'],
                'heatmap_size': results['heatmap_size'],
                'num_detected': results['num_detected'],
                'detection_params': results['detection_params'],
                'input_image': args.input_path,
                'model_path': args.model_path
            }
            
            json_path = os.path.join(args.output_dir, f"{image_name}_results.json")
            with open(json_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"💾 Detailed results saved to {json_path}")
    
    print(f"\n✅ Inference complete! Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()