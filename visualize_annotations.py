import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def load_annotation(label_path):
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

def draw_annotations(image_path, label_path, output_path):
    """Draw bounding box and keypoints on image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Load annotation
    annotation = load_annotation(label_path)
    if annotation is None:
        print(f"No annotation found in: {label_path}")
        return
    
    # Convert normalized coordinates to pixel coordinates
    center_x, center_y, width, height = annotation['bbox']
    center_x_px = int(center_x * w)
    center_y_px = int(center_y * h)
    width_px = int(width * w)
    height_px = int(height * h)
    
    # Calculate bounding box corners
    x1 = center_x_px - width_px // 2
    y1 = center_y_px - height_px // 2
    x2 = center_x_px + width_px // 2
    y2 = center_y_px + height_px // 2
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw class label
    class_name = 'pitch' if annotation['class_id'] == 1 else '0'
    cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Define colors for different visibility levels
    colors = {
        0: (128, 128, 128),  # Gray for not visible
        1: (0, 255, 0),      # Green for visible
        2: (0, 255, 255)     # Yellow for occluded
    }
    
    # Draw keypoints
    for i, (kx, ky, vis) in enumerate(annotation['keypoints']):
        if vis > 0:  # Only draw visible and occluded points
            x_px = int(kx * w)
            y_px = int(ky * h)
            color = colors.get(vis, (255, 255, 255))
            
            # Draw keypoint circle
            cv2.circle(img, (x_px, y_px), 4, color, -1)
            
            # Draw keypoint number
            cv2.putText(img, str(i), (x_px + 5, y_px - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Add legend
    legend_y = 30
    cv2.putText(img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.circle(img, (80, legend_y-5), 4, (0, 255, 0), -1)
    cv2.putText(img, "Visible", (90, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(img, (150, legend_y-5), 4, (0, 255, 255), -1)
    cv2.putText(img, "Occluded", (160, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(img, (230, legend_y-5), 4, (128, 128, 128), -1)
    cv2.putText(img, "Hidden", (240, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save result
    cv2.imwrite(output_path, img)
    print(f"Saved visualization: {output_path}")

def visualize_dataset_samples(data_dir, output_dir, num_samples=10):
    """Visualize random samples from the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    train_images = list(Path(data_dir).glob("train/images/*.jpg"))
    
    # Select random samples
    import random
    samples = random.sample(train_images, min(num_samples, len(train_images)))
    
    for img_path in samples:
        # Find corresponding label file
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        
        if label_path.exists():
            output_path = os.path.join(output_dir, f"viz_{img_path.name}")
            draw_annotations(str(img_path), str(label_path), output_path)
        else:
            print(f"Label file not found: {label_path}")

if __name__ == "__main__":
    # Visualize samples from the Brighton dataset
    data_dir = "brighton_data"
    output_dir = "visualizations"
    
    print("Creating visualizations of annotated soccer field keypoints...")
    visualize_dataset_samples(data_dir, output_dir, num_samples=10)
    print(f"Visualizations saved in '{output_dir}' directory")