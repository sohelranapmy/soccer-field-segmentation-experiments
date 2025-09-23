#!/usr/bin/env python3
"""
Batch generate heatmaps for entire SoccerNet 2023 calibration dataset
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

# Add project root to path
sys.path.append('/home/training-machine/Documents/brighton-project/No-Bells-Just-Whistles')

from utils.utils_heatmap import generate_gaussian_array_vectorized

def extract_keypoints_from_soccernet_annotations(annotations, target_size=(960, 540)):
    """Extract keypoints from SoccerNet line annotations"""
    w, h = target_size
    keypoints = {}
    kp_id = 1
    
    # Extract all points from line annotations
    for annotation_key, points in annotations.items():
        if isinstance(points, list) and len(points) > 0:
            for i, point in enumerate(points):
                if isinstance(point, dict) and 'x' in point and 'y' in point:
                    # Convert normalized coordinates to pixel coordinates
                    x_pixel = point['x'] * w
                    y_pixel = point['y'] * h
                    
                    # Include all points (even slightly outside frame for close_to_frame)
                    margin = 15
                    if -margin <= x_pixel <= w + margin and -margin <= y_pixel <= h + margin:
                        keypoints[kp_id] = {
                            'x': x_pixel,
                            'y': y_pixel,
                            'in_frame': (0 <= x_pixel <= w and 0 <= y_pixel <= h),
                            'close_to_frame': True,
                            'annotation_type': annotation_key,
                            'point_index': i
                        }
                        kp_id += 1
    
    # Extract line intersections for important field features
    intersections = extract_line_intersections(annotations, target_size)
    for intersection in intersections:
        keypoints[kp_id] = {
            'x': intersection['x'],
            'y': intersection['y'],
            'in_frame': (0 <= intersection['x'] <= w and 0 <= intersection['y'] <= h),
            'close_to_frame': True,
            'annotation_type': 'intersection',
            'point_index': 0
        }
        kp_id += 1
    
    return keypoints

def extract_line_intersections(annotations, target_size):
    """Extract important line intersections"""
    w, h = target_size
    intersections = []
    
    # Convert annotations to line segments
    line_segments = {}
    
    for line_name, points in annotations.items():
        if isinstance(points, list) and len(points) >= 2:
            for i in range(len(points) - 1):
                if isinstance(points[i], dict) and isinstance(points[i+1], dict):
                    p1 = (points[i]['x'] * w, points[i]['y'] * h)
                    p2 = (points[i+1]['x'] * w, points[i+1]['y'] * h)
                    
                    if line_name not in line_segments:
                        line_segments[line_name] = []
                    line_segments[line_name].append((p1, p2))
    
    # Find intersections between different line types
    penalty_lines = []
    sidelines = []
    goal_lines = []
    
    for line_name, segments in line_segments.items():
        line_lower = line_name.lower()
        if "rect" in line_lower or "box" in line_lower:
            penalty_lines.extend(segments)
        elif "side" in line_lower:
            sidelines.extend(segments)
        elif "goal" in line_lower:
            goal_lines.extend(segments)
    
    # Calculate useful intersections
    all_line_pairs = [
        (penalty_lines, sidelines),
        (penalty_lines, goal_lines),
        (sidelines, goal_lines)
    ]
    
    for lines1, lines2 in all_line_pairs:
        for seg1 in lines1:
            for seg2 in lines2:
                intersection = line_intersection(seg1, seg2)
                if intersection and 0 <= intersection[0] <= w and 0 <= intersection[1] <= h:
                    intersections.append({'x': intersection[0], 'y': intersection[1]})
    
    return intersections

def line_intersection(line1, line2):
    """Calculate intersection of two line segments"""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None

def process_single_sample(args):
    """Process a single sample and generate heatmap"""
    sample_id, input_dir, output_dir, target_size, sigma, down_ratio = args
    
    try:
        # Load image and annotations
        image_path = os.path.join(input_dir, f"{sample_id}.jpg")
        json_path = os.path.join(input_dir, f"{sample_id}.json")
        
        if not os.path.exists(image_path) or not os.path.exists(json_path):
            return False, f"Missing files for {sample_id}"
        
        # Load data
        image = Image.open(image_path)
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        
        # Extract keypoints
        keypoints = extract_keypoints_from_soccernet_annotations(annotations, target_size)
        
        if len(keypoints) == 0:
            return False, f"No keypoints extracted for {sample_id}"
        
        # Generate heatmaps at full resolution (no downsampling)
        num_channels = 58  # Standard number of channels
        heatmap_tensor = generate_gaussian_array_vectorized(
            num_channels, keypoints, target_size,
            down_ratio=1, sigma=sigma  # down_ratio=1 means same resolution as input
        )
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, f"{sample_id}_heatmap.npz")
        # np.save(heatmap_path, heatmap_tensor)
        np.savez_compressed(heatmap_path, heatmap_tensor)
        # Save metadata
        metadata = {
            'sample_id': sample_id,
            'original_image_size': image.size,
            'target_size': target_size,
            'num_keypoints': len(keypoints),
            'visible_keypoints': sum(1 for kp in keypoints.values() if kp['in_frame']),
            'heatmap_shape': heatmap_tensor.shape,
            'sigma': sigma,
            'down_ratio': down_ratio,
            'keypoint_types': list(set(kp['annotation_type'] for kp in keypoints.values()))
        }
        
        metadata_path = os.path.join(output_dir, f"{sample_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True, f"Processed {sample_id}: {len(keypoints)} keypoints, shape {heatmap_tensor.shape}"
        
    except Exception as e:
        return False, f"Error processing {sample_id}: {str(e)}"

def batch_generate_heatmaps(input_dir, output_dir, split_name, target_size=(960, 540), 
                           sigma=2.0, down_ratio=1, num_workers=4, max_samples=None):
    """Batch generate heatmaps for a dataset split"""
    
    print(f"\n=== Batch Generating Heatmaps for {split_name.upper()} ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Sigma: {sigma}, Down ratio: {down_ratio}")
    print(f"Workers: {num_workers}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all samples
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    sample_ids = [f.replace('.json', '') for f in json_files]
    sample_ids = sorted(sample_ids)
    
    if max_samples:
        sample_ids = sample_ids[:max_samples]
        print(f"Processing first {max_samples} samples for testing")
    
    print(f"Found {len(sample_ids)} samples to process")
    
    # Prepare arguments for multiprocessing
    process_args = [
        (sample_id, input_dir, output_dir, target_size, sigma, down_ratio)
        for sample_id in sample_ids
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
        'total_samples': len(sample_ids),
        'successful': successful,
        'failed': failed,
        'target_size': target_size,
        'sigma': sigma,
        'down_ratio': down_ratio,
        'output_directory': output_dir
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
    parser = argparse.ArgumentParser(description='Batch generate SoccerNet heatmaps')
    parser.add_argument('--dataset_root', default='/home/training-machine/Documents/brighton-project/soccer_net_data/calibration-2023',
                       help='Root directory of SoccerNet dataset')

    parser.add_argument('--target_size', default='960,540', help='Target image size (width,height)')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian sigma for heatmaps')
    parser.add_argument('--down_ratio', type=int, default=1, help='Downsampling ratio for heatmaps (1=same as image size)')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    parser.add_argument('--splits', default='train,valid', help='Dataset splits to process (comma-separated)')
    parser.add_argument('--max_samples', type=int, help='Maximum samples per split (for testing)')
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Parse splits
    splits = args.splits.split(',')
    
    print("=" * 60)
    print("SoccerNet 2023 Heatmap Batch Generation")
    print("=" * 60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Target size: {target_size}")
    print(f"Sigma: {args.sigma}")
    print(f"Down ratio: {args.down_ratio}")
    print(f"Workers: {args.workers}")
    print(f"Splits: {splits}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    
    total_successful = 0
    total_failed = 0
    
    # Process each split
    for split in splits:
        input_dir = os.path.join(args.dataset_root, split)
        output_dir = os.path.join(args.dataset_root, f"{split}_heatmaps")
        
        if not os.path.exists(input_dir):
            print(f"❌ Split directory not found: {input_dir}")
            continue
        
        successful, failed = batch_generate_heatmaps(
            input_dir=input_dir,
            output_dir=output_dir,
            split_name=split,
            target_size=target_size,
            sigma=args.sigma,
            down_ratio=args.down_ratio,
            num_workers=args.workers,
            max_samples=args.max_samples
        )
        
        total_successful += successful
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")
    print(f"Overall success rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
    print(f"Output directory: {args.output_root}")
    print("Ready for training! ✅")

if __name__ == '__main__':
    main()