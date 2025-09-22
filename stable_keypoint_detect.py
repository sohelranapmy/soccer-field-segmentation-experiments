import cv2
import yaml
import torch
import argparse
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as f
from tqdm import tqdm
from PIL import Image
import os
import copy
import math

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


class KeypointStabilityAnalyzer:
    def __init__(self, window_size=50,min_stability=0.3):
        self.detection_history = {}
        self.window_size = window_size
        self.min_stability = min_stability
        self.stability_scores = {}
        
    def update_detection_history(self, frame_keypoints):
        for kp_id in range(1, 31):
            if kp_id not in self.detection_history:
                self.detection_history[kp_id] = []
            
            detected = kp_id in frame_keypoints
            confidence = frame_keypoints[kp_id]['p'] if detected else 0.0
            position = (frame_keypoints[kp_id]['x'], frame_keypoints[kp_id]['y']) if detected else None
            
            self.detection_history[kp_id].append({
                'detected': detected,
                'confidence': confidence,
                'position': position
            })
            
            if len(self.detection_history[kp_id]) > self.window_size:
                self.detection_history[kp_id].pop(0)
    
    def calculate_stability_scores(self):
        for kp_id, history in self.detection_history.items():
            if len(history) < 10:  # Lowered from 5 to 3 for more coverage
                continue
                
            detection_rate = sum(1 for h in history if h['detected']) / len(history)
            
            # positions = [h['position'] for h in history if h['position']]
            # if len(positions) >= 3:
            #     x_coords = [p[0] for p in positions]
            #     y_coords = [p[1] for p in positions]
            #     position_stability = 1.0 / (1.0 + np.std(x_coords) + np.std(y_coords))
            # else:
            #     position_stability = 0.0
            
            # confidences = [h['confidence'] for h in history if h['detected']]
            # confidence_stability = np.mean(confidences) if confidences else 0.0
            
            # streak_score = self.calculate_streak_score(history)
            
            # self.stability_scores[kp_id] = {
            #     'detection_rate': detection_rate,
            #     'position_stability': position_stability,
            #     'confidence_stability': confidence_stability,
            #     'streak_score': streak_score,
            #     # 'overall': (detection_rate * 0.4 + position_stability * 0.3 + 
            #     #            confidence_stability * 0.2 + streak_score * 0.1)
            #     'overall': (detection_rate * 1.0 + position_stability * 0.0 + 
            #                confidence_stability * 0.0 + streak_score * 0.0)
            # }
            self.stability_scores[kp_id] = {
                'overall': detection_rate  # Simplified overall score
            }
    def calculate_streak_score(self, history):
        if not history:
            return 0.0
        
        streaks = []
        current_streak = 0
        
        for entry in history:
            if entry['detected']:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return np.mean(streaks) / len(history) if streaks else 0.0
    
    def get_stable_anchors(self, min_stability=0.5, max_anchors=5):
        # if not self.stability_scores:
        #     self.calculate_stability_scores()
        self.calculate_stability_scores()
        sorted_kps = sorted(self.stability_scores.items(), 
                           key=lambda x: x[1]['overall'], reverse=True)
        
        stable_anchors = []
        for kp_id, scores in sorted_kps:
            if scores['overall'] >= min_stability:# and len(stable_anchors) < max_anchors:
                stable_anchors.append(kp_id)
        
        return stable_anchors
    
    def get_adaptive_anchors(self, current_detections):
        stable_candidates = self.get_stable_anchors(min_stability=0.0)  # Changed from 0.0 to 0.1
        # print(f"Stable candidates: {stable_candidates}")
        current_stable = [kp for kp in stable_candidates if kp in current_detections]
        
        # if len(current_stable) < 2:
        #     fallback = [kp for kp, data in current_detections.items() 
        #                if data['p'] > 0.7]
        #     current_stable.extend(fallback[:3])
        
        return current_stable#[:5]


class AdaptiveRelativeMemory:
    def __init__(self):
        self.stability_analyzer = KeypointStabilityAnalyzer()
        self.relationships = {}
        self.relationship_counts = {}
        
    def learn_relationship(self, kp_a, kp_b, pos_a, pos_b):
        dx = pos_a['x'] - pos_b['x']
        dy = pos_a['y'] - pos_b['y']
        distance = math.sqrt(dx**2 + dy**2)
        
        rel_key = (kp_a, kp_b)
        
        if rel_key not in self.relationships:
            self.relationships[rel_key] = {
                'dx': dx, 'dy': dy, 'distance': distance, 'count': 1
            }
        else:
            count = self.relationships[rel_key]['count']
            self.relationships[rel_key]['dx'] = (self.relationships[rel_key]['dx'] * count + dx) / (count + 1)
            self.relationships[rel_key]['dy'] = (self.relationships[rel_key]['dy'] * count + dy) / (count + 1)
            self.relationships[rel_key]['distance'] = (self.relationships[rel_key]['distance'] * count + distance) / (count + 1)
            self.relationships[rel_key]['count'] += 1
    
    def update_frame(self, frame_keypoints):
        self.stability_analyzer.update_detection_history(frame_keypoints)
        
        # stable_anchors = self.stability_analyzer.get_adaptive_anchors(frame_keypoints)
        
        # for kp_id, kp_data in frame_keypoints.items():
        #     for anchor_id in stable_anchors:
        #         if anchor_id != kp_id and anchor_id in frame_keypoints:
        #             self.learn_relationship(kp_id, anchor_id, kp_data, frame_keypoints[anchor_id])
    
    def estimate_from_stable_anchors(self, missing_kp, stable_anchors, current_keypoints):
        estimates = []
        
        for anchor_id in stable_anchors:
            if anchor_id in current_keypoints:
                rel_key = (missing_kp, anchor_id)
                if rel_key in self.relationships and self.relationships[rel_key]['count'] >= 3:
                    rel_data = self.relationships[rel_key]
                    anchor_pos = current_keypoints[anchor_id]
                    
                    estimated_x = anchor_pos['x'] + rel_data['dx']
                    estimated_y = anchor_pos['y'] + rel_data['dy']
                    estimates.append((estimated_x, estimated_y))
        
        if len(estimates) >= 2:
            x_est = np.median([e[0] for e in estimates])
            y_est = np.median([e[1] for e in estimates])
            confidence = min(0.8, 0.4 + 0.1 * len(estimates))
            
            return {'x': x_est, 'y': y_est, 'p': confidence}
        
        return None
    
    def complete_missing_keypoints(self, current_keypoints):
        stable_anchors = self.stability_analyzer.get_adaptive_anchors(current_keypoints)
        completed = current_keypoints.copy()
        
        for missing_kp in range(1, 31):
            if missing_kp not in completed:
                estimated = self.estimate_from_stable_anchors(missing_kp, stable_anchors, completed)
                if estimated:
                    completed[missing_kp] = estimated
        
        return completed


def inference_keypoints_only(cam, frame, model, model_l, kp_threshold, line_threshold, memory_system=None):
    """Extract keypoints from frame and return final_dict with optional memory system"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    frame_tensor = f.to_tensor(frame_pil).float().unsqueeze(0)
    _, _, h_original, w_original = frame_tensor.size()
    frame_tensor = frame_tensor if frame_tensor.size()[-1] == 960 else transform2(frame_tensor)
    frame_tensor = frame_tensor.to(device)
    b, c, h, w = frame_tensor.size()

    with torch.no_grad():
        heatmaps = model(frame_tensor)
        heatmaps_l = model_l(frame_tensor)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    
    # Apply memory system if available
    # if memory_system is not None and len(kp_dict) > 0:
        # Update memory with current detections
        # memory_system.update_frame(kp_dict[0])
        
        # Complete missing keypoints using relative memory
        # completed_kp_dict = memory_system.complete_missing_keypoints(kp_dict[0])
        # kp_dict = [completed_kp_dict]
    
    final_dict = complete_keypoints(kp_dict, lines_dict, w=w, h=h, normalize=True)
    
    # Apply memory system if available - update after complete_keypoints
    if memory_system is not None:
        # Update memory with final processed keypoints
        memory_system.update_frame(final_dict[0])
    
    # Update camera calibration to denormalize coordinates back to pixel space
    cam.update(final_dict[0])
    
    return final_dict[0]


def draw_keypoints_on_frame(frame, keypoints_dict, show_confidence=True, min_confidence=0.0, memory_system=None, show_stability=False):
    """Draw keypoints on frame with confidence-based visualization and optional stability scores"""
    frame_with_kp = frame.copy()
    
    # Get stability scores if memory system is available
    stability_scores = {}
    stable_anchors = []
    if memory_system and show_stability:
        memory_system.stability_analyzer.calculate_stability_scores()
        stability_scores = memory_system.stability_analyzer.stability_scores
        stable_anchors = memory_system.stability_analyzer.get_adaptive_anchors(keypoints_dict)
    
    for kp_id, kp_data in keypoints_dict.items():
        if 'x' in kp_data and 'y' in kp_data:
            x, y = int(kp_data['x']), int(kp_data['y'])
            confidence = kp_data.get('p', 0.0)  # Get confidence score
            
            # Skip keypoints below minimum confidence
            if confidence < min_confidence:
                continue
            
            # Get stability score for this keypoint
            stability_score = 0.0
            is_stable_anchor = kp_id in stable_anchors
            if kp_id in stability_scores:
                stability_score = stability_scores[kp_id]['overall']
            
            # Color based on stability if showing stability, otherwise confidence
            if show_stability and kp_id in stability_scores:
                # Color based on stability: red (low) -> yellow -> green (high)
                if stability_score < 0.3:
                    color = (0, 0, 255)  # Red for low stability
                elif stability_score < 0.6:
                    color = (0, 165, 255)  # Orange for medium stability
                else:
                    color = (0, 255, 0)  # Green for high stability
                
                # Special color for stable anchors
                if is_stable_anchor:
                    color = (255, 0, 255)  # Magenta for stable anchors
            else:
                # Color based on confidence: red (low) -> yellow -> green (high)
                if confidence < 0.3:
                    color = (0, 0, 255)  # Red for low confidence
                elif confidence < 0.6:
                    color = (0, 165, 255)  # Orange for medium confidence
                else:
                    color = (0, 255, 0)  # Green for high confidence
            
            # Radius based on confidence (larger for higher confidence)
            radius = max(3, int(5 + confidence * 5))
            thickness = max(1, int(1 + confidence * 2))
            
            # Larger radius for stable anchors
            if is_stable_anchor:
                radius += 2
                thickness += 1
            
            # Draw circle for keypoint
            cv2.circle(frame_with_kp, (x, y), radius, color, thickness)
            
            # Draw keypoint ID, confidence, and stability info
            if show_confidence or show_stability:
                if show_stability:
                    if kp_id in stability_scores:
                        # Full stability score available
                        if is_stable_anchor:
                            label = f"{kp_id}:C{confidence:.2f}|S{stability_score:.2f}*"
                        else:
                            label = f"{kp_id}:C{confidence:.2f}|S{stability_score:.2f}"
                    # else:
                    #     label = f"{kp_id}:C{confidence:.2f}|S?"
                    # else:
                    #     # Fallback for keypoints without enough history or not in stability_scores
                    #     if memory_system and kp_id in memory_system.stability_analyzer.detection_history:
                    #         recent_detections = memory_system.stability_analyzer.detection_history[kp_id]
                    #         history_len = len(recent_detections)
                    #         if history_len > 0:
                    #             recent_rate = sum(1 for h in recent_detections if h['detected']) / len(recent_detections)
                    #             if history_len >= 10:
                    #                 # Should have stability score but doesn't - show recent rate with !
                    #                 label = f"{kp_id}:C{confidence:.2f}|S{recent_rate:.2f}!"
                    #             else:
                    #                 # Not enough history yet - show with ?
                    #                 label = f"{kp_id}:C{confidence:.2f}|S{recent_rate:.2f}?"
                    #         else:
                    #             label = f"{kp_id}:C{confidence:.2f}|S?"
                    #     else:
                    #         label = f"{kp_id}:C{confidence:.2f}|S?"
                elif show_confidence:
                    label = f"{kp_id}:{confidence:.2f}"
                else:
                    label = str(kp_id)
                
                # Background for text readability
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
                cv2.rectangle(frame_with_kp, (x + 8, y - 25), (x + 8 + text_size[0], y - 5), (0, 0, 0), -1)
                cv2.putText(frame_with_kp, label, (x + 8, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                cv2.putText(frame_with_kp, str(kp_id), (x + 8, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame_with_kp


def create_accumulated_heatmap_overlay(heatmap, frame_shape, colormap_type=cv2.COLORMAP_HOT, alpha=0.6):
    """Convert accumulated heatmap to overlay"""
    if heatmap.max() > 0:
        # Normalize to 0-255
        heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap_type)
        
        # Create alpha mask (make low values transparent)
        alpha_mask = (heatmap_norm > 25).astype(np.float32) * alpha
        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
        return heatmap_colored, alpha_mask
    else:
        return np.zeros((*frame_shape[:2], 3), dtype=np.uint8), np.zeros((*frame_shape[:2], 3), dtype=np.float32)


def create_keypoint_video(video_path, weights_kp, weights_line, output_path, 
                         frame_skip=1, kp_threshold=0.1486, line_threshold=0.3880, 
                         device="cuda:0", show_heatmap=True, show_keypoints=True,
                         show_confidence=True, min_confidence=0.0, heatmap_sigma=15,
                         use_memory_system=True, show_stability=False, max_frames=None):
    """
    Create output video with keypoints and/or accumulated heatmap visualization.
    
    Args:
        video_path (str): Input video path
        weights_kp (str): Keypoint model weights
        weights_line (str): Line model weights  
        output_path (str): Output video path
        frame_skip (int): Process every nth frame
        kp_threshold (float): Keypoint detection threshold
        line_threshold (float): Line detection threshold
        device (str): Device for inference
        show_heatmap (bool): Show accumulated heatmap overlay
        show_keypoints (bool): Show individual keypoints
        show_confidence (bool): Show confidence scores for keypoints
        min_confidence (float): Minimum confidence threshold for display
        heatmap_sigma (float): Gaussian sigma for heatmap smoothing
        use_memory_system (bool): Use adaptive relative memory for keypoint stability
        show_stability (bool): Show stability scores and highlight stable anchors
        max_frames (int): Maximum number of frames to process (None for all frames)
    """
    
    # Load models
    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state = torch.load(weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    global transform2
    transform2 = T.Resize((540, 960))

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Limit frames if max_frames is specified
    frames_to_process = total_frames // frame_skip
    if max_frames is not None:
        frames_to_process = min(frames_to_process, max_frames)
        actual_video_frames = frames_to_process * frame_skip
    else:
        actual_video_frames = total_frames
    
    # Adjust output FPS based on frame skip
    output_fps = max(1, fps // frame_skip)
    
    print(f"Input: {frame_width}x{frame_height}, {total_frames} frames, {fps} FPS")
    if max_frames is not None:
        print(f"Processing: {frames_to_process} frames (limited by max_frames={max_frames})")
    print(f"Output: {output_fps} FPS (every {frame_skip}th frame)")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))
    
    # Initialize heatmap accumulator if needed
    if show_heatmap:
        accumulated_heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    
    # Initialize memory system if enabled
    memory_system = AdaptiveRelativeMemory() if use_memory_system else None
    
    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    
    frame_count = 0
    processed_count = 0
    
    pbar = tqdm(total=frames_to_process, desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count<1500:
            frame_count+=1
            continue
        
        # Stop if we've processed enough frames
        if max_frames is not None and processed_count >= max_frames:
            break
            
        # Process only every frame_skip-th frame
        if frame_count % frame_skip == 0:
            try:
                # Get keypoints with memory system
                keypoints_dict = inference_keypoints_only(cam, frame, model, model_l, kp_threshold, line_threshold, memory_system)
                
                # Start with original frame
                output_frame = frame.copy()
                
                # Add keypoints to accumulated heatmap
                if show_heatmap:
                    for kp_id, kp_data in keypoints_dict.items():
                        if 'x' in kp_data and 'y' in kp_data:
                            x, y = int(kp_data['x']), int(kp_data['y'])
                            if 0 <= x < frame_width and 0 <= y < frame_height:
                                # Add Gaussian blob
                                y_coords, x_coords = np.ogrid[:frame_height, :frame_width]
                                gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * heatmap_sigma**2))
                                accumulated_heatmap += gaussian
                    
                    # Create heatmap overlay
                    heatmap_overlay, alpha_mask = create_accumulated_heatmap_overlay(
                        accumulated_heatmap, frame.shape
                    )
                    
                    # Blend heatmap with frame
                    output_frame = output_frame.astype(np.float32)
                    heatmap_overlay = heatmap_overlay.astype(np.float32)
                    output_frame = output_frame * (1 - alpha_mask) + heatmap_overlay * alpha_mask
                    output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)
                
                # Draw current frame keypoints
                if show_keypoints:
                    output_frame = draw_keypoints_on_frame(output_frame, keypoints_dict, show_confidence, min_confidence, memory_system, show_stability=show_stability)
                
                # Add frame info with confidence statistics and stability info
                if len(keypoints_dict) > 0:
                    confidences = [kp_data.get('p', 0.0) for kp_data in keypoints_dict.values() if 'p' in kp_data]
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    max_confidence = max(confidences) if confidences else 0.0
                    
                    # Add memory system info with detailed stability statistics
                    if memory_system:
                        memory_system.stability_analyzer.calculate_stability_scores()
                        stable_anchors = memory_system.stability_analyzer.get_adaptive_anchors(keypoints_dict)
                        stability_scores = memory_system.stability_analyzer.stability_scores
                        
                        # Calculate average stability score for detected keypoints
                        detected_stabilities = [stability_scores[kp_id]['overall'] 
                                              for kp_id in keypoints_dict.keys() 
                                              if kp_id in stability_scores]
                        avg_stability = np.mean(detected_stabilities) if detected_stabilities else 0.0
                        
                        stability_info = f", Anchors: {len(stable_anchors)}, Avg Stab: {avg_stability:.3f}"
                    else:
                        stability_info = ""
                    
                    info_text = f"Frame: {processed_count+1}, KPs: {len(keypoints_dict)}, Avg Conf: {avg_confidence:.3f}, Max: {max_confidence:.3f}{stability_info}"
                else:
                    info_text = f"Frame: {processed_count+1}, Keypoints: 0"
                
                # Background for text readability
                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(output_frame, (5, 5), (15 + text_size[0], 35), (0, 0, 0), -1)
                cv2.putText(output_frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 2)
                
                # Write frame
                out.write(output_frame)
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write original frame on error
                out.write(frame)
                processed_count += 1
                pbar.update(1)
        
        frame_count += 1
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"Output video saved: {output_path}")
    print(f"Processed {processed_count} frames from {total_frames} total frames")




def main():
    parser = argparse.ArgumentParser(description="Create video with keypoint visualization")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--weights_kp", type=str, required=True, help="Keypoint model weights")
    parser.add_argument("--weights_line", type=str, required=True, help="Line model weights")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every nth frame")
    parser.add_argument("--kp_threshold", type=float, default=0.1486, help="Keypoint threshold")
    parser.add_argument("--line_threshold", type=float, default=0.3880, help="Line threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--mode", type=str, choices=['heatmap', 'trails', 'keypoints'], default='heatmap',
                       help="Visualization mode: heatmap (accumulated), trails, or keypoints only")
    parser.add_argument("--trail_length", type=int, default=30, help="Trail length for trails mode")
    parser.add_argument("--heatmap_sigma", type=float, default=15, help="Gaussian sigma for heatmap")
    parser.add_argument("--show_confidence", action='store_true', default=True, help="Show confidence scores")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Minimum confidence threshold for display")
    parser.add_argument("--hide_confidence", action='store_true', help="Hide confidence scores (show only IDs)")
    parser.add_argument("--disable_memory", action='store_true', help="Disable adaptive relative memory system")
    parser.add_argument("--show_stability", action='store_true', help="Show stability scores and highlight stable anchors")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process (None for all frames)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    try:
        global device
        device = args.device
        
        show_conf = args.show_confidence and not args.hide_confidence
        use_memory = not args.disable_memory
        show_stab = args.show_stability
        

        create_keypoint_video(
            args.video, args.weights_kp, args.weights_line, args.output,
            args.frame_skip, args.kp_threshold, args.line_threshold, 
            args.device, show_heatmap=False, show_keypoints=True,
            show_confidence=show_conf, min_confidence=args.min_confidence,
            use_memory_system=use_memory, show_stability=show_stab, max_frames=args.max_frames
        )

        
        print("Video processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()