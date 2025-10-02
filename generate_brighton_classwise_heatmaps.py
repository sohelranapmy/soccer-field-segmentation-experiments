#!/usr/bin/env python3
"""
Batch generate heatmaps for Brighton soccer field keypoint dataset
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# Add project root to path
sys.path.append('/home/training-machine/Documents/brighton-project/No-Bells-Just-Whistles')

from utils.utils_heatmap import generate_gaussian_array_vectorized

def load_brighton_annotation(label_path):
    """Load YOLO keypoint annotation from text file"""
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    
    if not line:
        return None
    
    parts = line.split()
    class_id = int(parts[0])
    center_x, center_y, width, height = map(float, parts[1:5])
    
    # Extract keypoints (32 keypoints, 3 values each: x, y, visibility)
    keypoints = []
    for i in range(5, len(parts), 3):
        if i + 2 < len(parts):
            x, y, vis = float(parts[i]), float(parts[i+1]), int(parts[i+2])
            keypoints.append([x, y, vis])
    
    return {
        'class_id': class_id,
        'bbox': [center_x, center_y, width, height],
        'keypoints': keypoints
    }

def extract_keypoints_from_brighton_annotations(annotation, image_size, rescale_size=None):
    """Extract keypoints from Brighton YOLO format annotations"""
    if annotation is None:
        return []
    
    w, h = image_size
    keypoints = []
    
    # Calculate scale factors if rescaling
    if rescale_size is not None:
        scale_x = rescale_size[0] / w
        scale_y = rescale_size[1] / h
        target_w, target_h = rescale_size
    else:
        scale_x = scale_y = 1.0
        target_w, target_h = w, h
    
    # Convert normalized keypoints to pixel coordinates
    for i, (kx, ky, vis) in enumerate(annotation['keypoints']):
        if vis > 0:  # Only include visible and occluded points
            x_pixel = kx * w * scale_x
            y_pixel = ky * h * scale_y
            keypoints.append({
                'x': x_pixel,
                'y': y_pixel,
                'visibility': vis,
                'keypoint_id': i
            })
    
    return keypoints

def generate_keypoint_connections(keypoints):
    """Generate line connections between related keypoints based on soccer field structure"""
    connections = []
    
    # Define standard soccer field keypoint connections based on typical field structure
    # These indices are estimates based on common soccer field annotation patterns
    field_connections = [
        # Penalty box corners (assuming first 8 points are penalty areas)
        (0, 1), (1, 2), (2, 3), (3, 0),  # Left penalty box
        (4, 5), (5, 6), (6, 7), (7, 4),  # Right penalty box
        
        # Goal area connections (assuming next 8 points are goal areas)
        (8, 9), (9, 10), (10, 11), (11, 8),   # Left goal area
        (12, 13), (13, 14), (14, 15), (15, 12), # Right goal area
        
        # Sideline connections (assuming next points are field boundaries)
        (16, 17), (17, 18), (18, 19), (19, 16), # Outer field boundary
        
        # Center circle and line connections
        (20, 21), (22, 23), (24, 25), (26, 27),
        
        # Additional field markings
        (28, 29), (30, 31)
    ]
    
    # Create keypoint lookup by ID
    keypoint_dict = {kp['keypoint_id']: kp for kp in keypoints}
    
    # Generate connections where both keypoints exist and are visible
    for start_id, end_id in field_connections:
        if start_id in keypoint_dict and end_id in keypoint_dict:
            start_kp = keypoint_dict[start_id]
            end_kp = keypoint_dict[end_id]
            
            connections.append({
                'x1': start_kp['x'],
                'y1': start_kp['y'],
                'x2': end_kp['x'],
                'y2': end_kp['y'],
                'start_id': start_id,
                'end_id': end_id
            })
    
    return connections

def generate_classwise_keypoint_heatmaps(keypoints, image_size, sigma=2.0, num_classes=32):
    """Generate separate Gaussian heatmaps for each keypoint class + background class"""
    w, h = image_size  # width, height
    # Add 1 extra class for background (index 0), keypoint classes start from index 1
    total_classes = num_classes + 1
    heatmaps = np.zeros((total_classes, h, w), dtype=np.float32)
    
    # Background class (index 0) remains all zeros - no need to modify
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:h, :w]
    
    for keypoint in keypoints:
        x, y = keypoint['x'], keypoint['y']
        keypoint_id = keypoint['keypoint_id']
        # Shift keypoint classes to start from index 1 (background is index 0)
        class_id = keypoint_id + 1
        
        # Create Gaussian for keypoint
        if 0 <= x < w and 0 <= y < h and 1 <= class_id < total_classes:
            gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
            
            # Weight by visibility (visible points get full weight, occluded get partial)
            weight = 1.0 if keypoint['visibility'] == 1 else 0.7
            heatmaps[class_id] = np.maximum(heatmaps[class_id], gaussian * weight)
    
    return heatmaps

def generate_keypoint_heatmap_gaussian(keypoints, image_size, sigma=2.0):
    """Generate combined Gaussian heatmap on keypoint locations (for backward compatibility)"""
    w, h = image_size  # width, height
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:h, :w]
    
    for keypoint in keypoints:
        x, y = keypoint['x'], keypoint['y']
        
        # Create Gaussian for keypoint
        if 0 <= x < w and 0 <= y < h:
            gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
            
            # Weight by visibility (visible points get full weight, occluded get partial)
            weight = 1.0 if keypoint['visibility'] == 1 else 0.7
            heatmap = np.maximum(heatmap, gaussian * weight)
    
    return heatmap

def generate_line_heatmap_gaussian(connections, image_size, sigma=2.0):
    """Generate Gaussian heatmap on start and end points of field lines"""
    w, h = image_size  # width, height
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:h, :w]
    
    for connection in connections:
        x1, y1 = connection['x1'], connection['y1']
        x2, y2 = connection['x2'], connection['y2']
        
        # Create Gaussian for start point
        if 0 <= x1 < w and 0 <= y1 < h:
            gaussian1 = np.exp(-((x_coords - x1)**2 + (y_coords - y1)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian1)
        
        # Create Gaussian for end point
        if 0 <= x2 < w and 0 <= y2 < h:
            gaussian2 = np.exp(-((x_coords - x2)**2 + (y_coords - y2)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian2)
    
    return heatmap

def generate_line_heatmap(connections, image_size, line_width=2):
    """Generate binary heatmap from line connections"""
    w, h = image_size  # width, height
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for connection in connections:
        x1, y1 = int(connection['x1']), int(connection['y1'])
        x2, y2 = int(connection['x2']), int(connection['y2'])
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        
        # Draw line using cv2
        cv2.line(heatmap, (x1, y1), (x2, y2), 1.0, thickness=line_width)
    
    return heatmap

def create_classwise_heatmap_overlay(image, classwise_heatmaps, keypoints, alpha=0.5):
    """Create an overlay of all class heatmaps on original image with class IDs"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Resize image to match heatmap dimensions if needed
    h, w = classwise_heatmaps.shape[1], classwise_heatmaps.shape[2]
    if image_np.shape[:2] != (h, w):
        image_np = cv2.resize(image_np, (w, h))
    
    # Ensure image is in RGB format
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = image_np
    else:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Combine all class heatmaps (excluding background class at index 0)
    combined_heatmap = np.max(classwise_heatmaps[1:], axis=0)
    
    # Normalize heatmap to 0-1 range
    if combined_heatmap.max() > 0:
        heatmap_normalized = (combined_heatmap - combined_heatmap.min()) / (combined_heatmap.max() - combined_heatmap.min() + 1e-8)
    else:
        heatmap_normalized = combined_heatmap
    
    # Apply colormap to heatmap (using 'jet' colormap)
    heatmap_colored = cm.jet(heatmap_normalized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend the images
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
    
    # Add class ID text annotations for visible keypoints
    for keypoint in keypoints:
        x, y = int(keypoint['x']), int(keypoint['y'])
        class_id = keypoint['keypoint_id']
        
        if 0 <= x < w and 0 <= y < h:
            # Add class ID text
            cv2.putText(overlay, str(class_id), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(overlay, str(class_id), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return overlay

def create_heatmap_overlay(image, heatmap_tensor, alpha=0.5):
    """Create an overlay of heatmap on original image (for backward compatibility)"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Resize image to match heatmap dimensions if needed
    if image_np.shape[:2] != heatmap_tensor.shape:
        image_np = cv2.resize(image_np, (heatmap_tensor.shape[1], heatmap_tensor.shape[0]))
    
    # Normalize heatmap to 0-1 range
    heatmap_normalized = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-8)
    
    # Apply colormap to heatmap (using 'jet' colormap)
    heatmap_colored = cm.jet(heatmap_normalized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Ensure image is in RGB format
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_rgb = image_np
    else:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Blend the images
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def process_single_sample(args):
    """Process a single sample and generate heatmap"""
    image_path, label_path, output_dir, sigma, heatmap_type, rescale_size, rescaled_images_dir = args
    
    try:
        # Extract sample ID from filename
        sample_id = Path(image_path).stem
        
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            return False, f"Missing files for {sample_id}"
        
        # Load image and annotations
        image = Image.open(image_path)
        annotation = load_brighton_annotation(label_path)
        
        if annotation is None:
            return False, f"No annotation found for {sample_id}"
        
        # Get original image size
        original_size = image.size  # (width, height)
        
        # Apply rescaling if specified
        rescaled_image_path = None
        if rescale_size is not None:
            image = image.resize(rescale_size, Image.Resampling.LANCZOS)
            image_size = rescale_size
            
            # Save rescaled image to separate folder
            rescaled_image_path = os.path.join(rescaled_images_dir, f"{sample_id}.jpg")
            image.save(rescaled_image_path, quality=95)
        else:
            image_size = original_size
        
        # Extract keypoints
        keypoints = extract_keypoints_from_brighton_annotations(annotation, original_size, rescale_size)
        
        if len(keypoints) == 0:
            return False, f"No keypoints extracted for {sample_id}"
        
        # Generate classwise heatmaps for all keypoints (32 keypoint classes + 1 background class)
        classwise_heatmaps = generate_classwise_keypoint_heatmaps(keypoints, image_size, sigma=sigma, num_classes=32)
        
        # Also generate traditional combined heatmap for backward compatibility
        if heatmap_type == 'keypoints':
            heatmap_tensor = generate_keypoint_heatmap_gaussian(keypoints, image_size, sigma=sigma)
        elif heatmap_type == 'lines':
            connections = generate_keypoint_connections(keypoints)
            if len(connections) == 0:
                return False, f"No line connections generated for {sample_id}"
            heatmap_tensor = generate_line_heatmap_gaussian(connections, image_size, sigma=sigma)
        else:  # combined
            # Generate both keypoint and line heatmaps and combine
            keypoint_heatmap = generate_keypoint_heatmap_gaussian(keypoints, image_size, sigma=sigma)
            connections = generate_keypoint_connections(keypoints)
            if len(connections) > 0:
                line_heatmap = generate_line_heatmap_gaussian(connections, image_size, sigma=sigma)
                heatmap_tensor = np.maximum(keypoint_heatmap, line_heatmap)
            else:
                heatmap_tensor = keypoint_heatmap
        
        # Save classwise heatmaps
        heatmap_path = os.path.join(output_dir, f"{sample_id}_heatmap.npz")
        np.savez_compressed(heatmap_path, 
                           classwise_heatmaps=classwise_heatmaps,
                           combined_heatmap=heatmap_tensor,
                           keypoint_classes=[kp['keypoint_id'] for kp in keypoints])
        
        # Create and save classwise heatmap overlay with class IDs
        overlay_image = create_classwise_heatmap_overlay(image, classwise_heatmaps, keypoints, alpha=0.4)
        overlay_path = os.path.join(output_dir, f"{sample_id}_overlay.jpg")
        overlay_pil = Image.fromarray(overlay_image)
        overlay_pil.save(overlay_path, quality=85)
        
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'original_image_path': image_path,
            'original_image_size': original_size,
            'rescaled_image_path': rescaled_image_path,
            'rescaled_image_size': image_size if rescale_size is not None else None,
            'heatmap_size': image_size,
            'num_keypoints': len(keypoints),
            'classwise_heatmaps_shape': classwise_heatmaps.shape,
            'combined_heatmap_shape': heatmap_tensor.shape,
            'heatmap_type': heatmap_type,
            'sigma': sigma,
            'class_id': annotation['class_id'],
            'keypoint_ids': [kp['keypoint_id'] for kp in keypoints],
            'keypoint_classes_present': sorted(list(set([kp['keypoint_id'] + 1 for kp in keypoints]))),  # Shifted by 1 due to background class
            'background_class_index': 0,
            'keypoint_class_range': [1, 32],  # Keypoint classes are now 1-32, background is 0
            'total_classes': 33,  # 32 keypoint classes + 1 background class
            'rescale_applied': rescale_size is not None,
            'has_classwise_heatmaps': True
        }
        
        if heatmap_type in ['lines', 'combined']:
            connections = generate_keypoint_connections(keypoints)
            metadata['num_connections'] = len(connections)
        
        metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"Processed {sample_id}: {len(keypoints)} keypoints, classwise shape {classwise_heatmaps.shape}, combined shape {heatmap_tensor.shape}"
        
    except Exception as e:
        return False, f"Error processing {sample_id}: {str(e)}"

def batch_generate_heatmaps(data_dir, split_name, sigma=2.0, heatmap_type='combined', num_workers=4, max_samples=None, rescale_size=None):
    """Batch generate heatmaps for a Brighton dataset split"""
    
    print(f"\n=== Batch Generating Brighton Heatmaps for {split_name.upper()} ===")
    print(f"Data directory: {data_dir}")
    if rescale_size is not None:
        print(f"Rescaling images to: {rescale_size[0]}x{rescale_size[1]}")
    else:
        print(f"Using original image sizes for heatmap generation")
    print(f"Sigma: {sigma}, Heatmap type: {heatmap_type}")
    print(f"Workers: {num_workers}")
    
    # Create output directories
    if rescale_size is not None:
        output_dir = os.path.join(data_dir, f"{split_name}_rescaled_images_heatmaps")
    else:
        output_dir = os.path.join(data_dir, f"{split_name}_heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create rescaled images directory if rescaling is enabled
    rescaled_images_dir = None
    if rescale_size is not None:
        rescaled_images_dir = os.path.join(data_dir, f"{split_name}_rescaled_images")
        os.makedirs(rescaled_images_dir, exist_ok=True)
        print(f"Rescaled images will be saved to: {rescaled_images_dir}")
    
    # Find all image-label pairs
    images_dir = os.path.join(data_dir, split_name, "images")
    labels_dir = os.path.join(data_dir, split_name, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ Required directories not found: {images_dir} or {labels_dir}")
        return 0, 0
    
    # Find matching image-label pairs
    image_files = list(Path(images_dir).glob("*.jpg"))
    valid_pairs = []
    
    for img_path in image_files:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            valid_pairs.append((str(img_path), str(label_path)))
    
    if max_samples:
        valid_pairs = valid_pairs[:max_samples]
        print(f"Processing first {max_samples} samples for testing")
    
    print(f"Found {len(valid_pairs)} valid image-label pairs to process")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (img_path, label_path, output_dir, sigma, heatmap_type, rescale_size, rescaled_images_dir)
        for img_path, label_path in valid_pairs
    ]
    
    # Process samples
    successful = 0
    failed = 0
    
    if num_workers > 1:
        # Multiprocessing
        print(f"Using {num_workers} workers for parallel processing...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_sample, process_args),
                total=len(process_args),
                desc=f"Processing {split_name}"
            ))
    else:
        # Single-threaded
        print("Using single-threaded processing...")
        results = []
        for args in tqdm(process_args, desc=f"Processing {split_name}"):
            results.append(process_single_sample(args))
    
    # Count results
    for success, message in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(f"❌ {message}")
    
    # Save processing summary
    summary = {
        'split_name': split_name,
        'total_samples': len(valid_pairs),
        'successful': successful,
        'failed': failed,
        'rescale_size': rescale_size,
        'uses_original_image_sizes': rescale_size is None,
        'sigma': sigma,
        'heatmap_type': heatmap_type,
        'output_directory': output_dir,
        'rescaled_images_directory': rescaled_images_dir
    }
    
    summary_path = os.path.join(output_dir, f"{split_name}_processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ {split_name.upper()} Processing Complete:")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"   Summary saved to: {summary_path}")
    
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description='Batch generate Brighton soccer field heatmaps')
    parser.add_argument('--data_dir', default='brighton_data',
                       help='Brighton dataset directory')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian sigma for heatmaps')
    parser.add_argument('--heatmap_type', choices=['keypoints', 'lines', 'combined'], default='combined',
                       help='Type of heatmap to generate')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--splits', default='train,valid,test', help='Dataset splits to process (comma-separated)')
    parser.add_argument('--max_samples', type=int, help='Maximum samples per split (for testing)')
    parser.add_argument('--rescale_size', type=str, help='Rescale images to specified size (format: WIDTHxHEIGHT, e.g., 640x480)')
    
    args = parser.parse_args()
    
    # Parse splits
    splits = args.splits.split(',')
    
    # Parse rescale size
    rescale_size = None
    if args.rescale_size:
        try:
            width, height = map(int, args.rescale_size.split('x'))
            rescale_size = (width, height)
        except ValueError:
            print(f"❌ Invalid rescale_size format: {args.rescale_size}. Use format: WIDTHxHEIGHT (e.g., 640x480)")
            return
    
    print("=" * 60)
    print("Brighton Soccer Field Heatmap Batch Generation")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    if rescale_size is not None:
        print(f"Rescaling images to: {rescale_size[0]}x{rescale_size[1]}")
    else:
        print(f"Using original image sizes for heatmaps")
    print(f"Sigma: {args.sigma}")
    print(f"Heatmap type: {args.heatmap_type}")
    print(f"Workers: {args.workers}")
    print(f"Splits: {splits}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    
    total_successful = 0
    total_failed = 0
    
    # Process each split
    for split in splits:
        if not os.path.exists(os.path.join(args.data_dir, split)):
            print(f"❌ Split directory not found: {os.path.join(args.data_dir, split)}")
            continue
        
        successful, failed = batch_generate_heatmaps(
            data_dir=args.data_dir,
            split_name=split,
            sigma=args.sigma,
            heatmap_type=args.heatmap_type,
            num_workers=args.workers,
            max_samples=args.max_samples,
            rescale_size=rescale_size
        )
        
        total_successful += successful
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    if total_successful + total_failed > 0:
        print(f"Overall success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
    print(f"Heatmaps saved in respective split directories under {args.data_dir}")
    print("Ready for training! ✅")

if __name__ == '__main__':
    main()