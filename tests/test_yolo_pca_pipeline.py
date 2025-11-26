#!/usr/bin/env python3
"""
Test script for simplified YOLO → PCA → Rotate → YOLO pipeline.

This tests the proposed simplified detection pipeline:
1. YOLO to find strip in original image
2. PCA to detect rotation angle
3. Rotate entire image
4. YOLO to find strip again in rotated image
5. Show results in web review interface
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.detection.yolo_detector import YoloDetector
from services.utils.debug import DebugContext
from utils.image_loader import load_image

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not available. PCA rotation detection will not work.")
    print("Install with: pip install scikit-learn")


def detect_rotation_with_pca(image: np.ndarray, bbox: dict) -> float:
    """
    Detect rotation angle using PCA on edge points.
    
    Args:
        image: Full image (BGR format)
        bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        Rotation angle in degrees (positive = counterclockwise)
    """
    if not SKLEARN_AVAILABLE:
        return 0.0
    
    # Crop to bbox region
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    # Expand bbox to include edges that might be slightly outside YOLO detection
    # Expand more in horizontal direction (left/right) since strip edges are vertical
    h, w = image.shape[:2]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expand_x = int(bbox_width * 0.10)  # 10% expansion horizontally (left/right)
    expand_y = int(bbox_height * 0.05)  # 5% expansion vertically (top/bottom)
    
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get foreground pixels (the strip itself)
    # The strip is white/high contrast, so we want to threshold to get the strip region
    # Use Otsu's method for automatic threshold selection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Get all foreground pixel coordinates (where binary > 0)
    # These are the pixels that belong to the strip
    foreground_points = np.column_stack(np.where(binary > 0))
    
    if len(foreground_points) < 100:
        # Not enough foreground pixels, try adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        foreground_points = np.column_stack(np.where(binary > 0))
    
    if len(foreground_points) < 100:
        # Still not enough points, return 0
        return 0.0, {
            'foreground_points_count': len(foreground_points),
            'crop_size': crop.shape[:2],
            'method': 'threshold_failed'
        }
    
    # Center the points
    center = foreground_points.mean(axis=0)
    centered_points = foreground_points - center
    
    # Parameters going into PCA
    pca_params = {
        'foreground_points_count': len(foreground_points),
        'threshold_method': 'otsu',
        'crop_size': crop.shape[:2],
        'bbox_original': {'x1': bbox['x1'], 'y1': bbox['y1'], 'x2': bbox['x2'], 'y2': bbox['y2']},
        'bbox_expanded': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        'expansion': {'x': expand_x, 'y': expand_y},
        'centered_points_shape': centered_points.shape,
        'center': center.tolist()
    }
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    
    # Get first principal component (direction of maximum variance)
    # This should align with the strip's main axis
    pc1 = pca.components_[0]
    
    # Calculate angle from horizontal
    # pc1 is in (row, col) format = (y, x) in image coordinates
    # For image coordinates: x increases right, y increases down
    # atan2(y, x) gives angle from positive x-axis
    # But we want angle from horizontal (x-axis)
    angle_rad = np.arctan2(pc1[0], pc1[1])  # atan2(row, col) = atan2(y, x)
    angle_deg = np.degrees(angle_rad)
    
    # Check aspect ratio to determine expected orientation
    crop_h, crop_w = crop.shape[:2]
    aspect_ratio = crop_h / crop_w if crop_w > 0 else 0
    
    # For a vertical strip (tall, aspect_ratio > 1):
    # - Edges should be vertical (angle close to ±90°)
    # - If angle is 89°, strip is 1° off vertical, rotate by -1°
    # - If angle is 91°, strip is -1° off vertical, rotate by +1°
    # - So: rotation = 90° - angle (for vertical strips)
    
    # For a horizontal strip (wide, aspect_ratio < 1):
    # - Edges should be horizontal (angle close to 0°)
    # - If angle is 1°, rotate by -1°
    # - So: rotation = -angle (for horizontal strips)
    
    if aspect_ratio > 1:
        # Vertical strip: PCA angle should be close to ±90°
        # Convert edge angle to rotation needed to make it vertical
        # If edge is at 89° (1° off from 90°), we need +1° rotation (counterclockwise)
        # If edge is at 91° (-1° off from 90°), we need -1° rotation (clockwise)
        # But wait: if PCA detects 89°, the strip is rotated 1° clockwise from vertical
        # To correct: rotate counterclockwise by +1°
        # So: rotation = 90 - angle, but we need to check the sign
        # Actually, if angle is 89° (1° less than 90°), strip is rotated 1° clockwise
        # To correct clockwise rotation, we rotate counterclockwise: +1°
        rotation_needed = angle_deg - 90.0  # Flip the sign
    else:
        # Horizontal strip: PCA angle should be close to 0°
        # If edge is at 1°, rotate by -1° to make it 0°
        rotation_needed = -angle_deg
    
    # Clamp to reasonable range (-45° to +45°)
    # Large rotations suggest the strip is actually horizontal/vertical
    if abs(rotation_needed) > 45:
        # If rotation is > 45°, the strip might be in wrong orientation
        # Try the other orientation
        if aspect_ratio > 1:
            # Maybe it's actually horizontal?
            rotation_needed = -angle_deg
        else:
            # Maybe it's actually vertical?
            rotation_needed = angle_deg - 90.0
    
    # Clamp final result to reasonable range
    rotation_needed = max(-45.0, min(45.0, rotation_needed))
    
    # Add PCA results to params
    pca_params.update({
        'pca_components': pca.components_.tolist(),
        'pca_explained_variance': pca.explained_variance_.tolist(),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pc1': pca.components_[0].tolist(),
        'angle_rad': float(angle_rad),
        'angle_deg': float(angle_deg),
        'aspect_ratio': float(aspect_ratio),
        'rotation_needed': float(rotation_needed)
    })
    
    return rotation_needed, pca_params


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate entire image by angle degrees.
    
    Args:
        image: Input image (BGR format)
        angle: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated image
    """
    # Apply rotation even for small angles to see the effect
    # (removed the 0.1° threshold to allow small corrections)
    if abs(angle) < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to fit rotated image
    angle_rad = np.radians(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust translation to center the rotated image
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # Rotate image
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White background
    )
    
    return rotated


def draw_bbox(image: np.ndarray, bbox: dict, color: tuple, label: str, thickness: int = 3) -> np.ndarray:
    """Draw bounding box on image."""
    vis = image.copy()
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    return vis


def process_image(image_path: str, debug: DebugContext):
    """
    Process image through YOLO → PCA → Rotate → YOLO pipeline.
    
    Args:
        image_path: Path to input image
        debug: DebugContext for logging steps
    
    Returns:
        Tuple of (result_dict, final_comparison_image)
    """
    # Load image
    image = load_image(image_path)
    h, w = image.shape[:2]
    
    # Log input image
    debug.add_step(
        '00_input',
        'Input Image',
        image,
        {'width': w, 'height': h},
        'Original input image'
    )
    
    # Initialize YOLO detector
    yolo = YoloDetector()
    if not yolo.is_available():
        debug.add_step(
            'error',
            'YOLO Not Available',
            image,
            {'error': 'YOLO detector not available'},
            'YOLO model failed to load'
        )
        return None
    
    # Step 1: YOLO detection on original image
    result1 = yolo.detect_strip(image)
    
    if not result1.get('success'):
        debug.add_step(
            'error',
            'YOLO Detection Failed',
            image,
            {'error': result1.get('error', 'Unknown error')},
            'YOLO detection failed on original image'
        )
        return None, image
    
    bbox1 = result1['bbox']
    conf1 = result1['confidence']
    
    # Visualize step 1
    vis1 = draw_bbox(image, bbox1, (0, 255, 0), f'YOLO 1 (conf={conf1:.2f})')
    
    debug.add_step(
        '01_yolo1',
        'YOLO Detection (Original)',
        vis1,
        {
            'bbox': bbox1,
            'confidence': conf1,
            'width': bbox1['x2'] - bbox1['x1'],
            'height': bbox1['y2'] - bbox1['y1']
        },
        f'YOLO detected strip with confidence {conf1:.3f}'
    )
    
    # Step 2: PCA rotation detection
    rotation_angle, pca_params = detect_rotation_with_pca(image, bbox1)
    
    # Visualize PCA detection (show foreground pixels used)
    # Show the expanded region that was actually used for PCA
    bbox_width = bbox1['x2'] - bbox1['x1']
    bbox_height = bbox1['y2'] - bbox1['y1']
    expand_x = int(bbox_width * 0.10)  # 10% horizontal expansion
    expand_y = int(bbox_height * 0.05)  # 5% vertical expansion
    expanded_x1 = max(0, bbox1['x1'] - expand_x)
    expanded_y1 = max(0, bbox1['y1'] - expand_y)
    expanded_x2 = min(image.shape[1], bbox1['x2'] + expand_x)
    expanded_y2 = min(image.shape[0], bbox1['y2'] + expand_y)
    
    crop_expanded = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    gray = cv2.cvtColor(crop_expanded, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    vis_pca = image.copy()
    vis_pca[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = binary_colored
    # Draw original YOLO bbox in green
    cv2.rectangle(vis_pca, (bbox1['x1'], bbox1['y1']), (bbox1['x2'], bbox1['y2']), (0, 255, 0), 2)
    # Draw expanded bbox in yellow
    cv2.rectangle(vis_pca, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 255, 0), 2)
    cv2.putText(vis_pca, f'PCA Angle: {rotation_angle:.2f}°', (expanded_x1, expanded_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    debug.add_step(
        '02_pca',
        'PCA Rotation Detection',
        vis_pca,
        {
            'rotation_angle': rotation_angle,
            'pca_params': pca_params,
            'bbox_used': bbox1
        },
        f'PCA detected rotation angle of {rotation_angle:.2f}°'
    )
    
    # Step 3: Rotate entire image
    rotated_image = rotate_image(image, rotation_angle)
    rh, rw = rotated_image.shape[:2]
    
    vis_rotation = rotated_image.copy()
    # Draw a reference line to show rotation
    h_rot, w_rot = rotated_image.shape[:2]
    center_x, center_y = w_rot // 2, h_rot // 2
    # Draw vertical reference line
    cv2.line(vis_rotation, (center_x, 0), (center_x, h_rot), (0, 255, 0), 2)
    cv2.putText(vis_rotation, f'Rotated {rotation_angle:.3f}°', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    debug.add_step(
        '03_rotated',
        'Rotated Image',
        vis_rotation,
        {
            'rotation_angle': rotation_angle,
            'original_size': {'width': w, 'height': h},
            'rotated_size': {'width': rw, 'height': rh}
        },
        f'Image rotated by {rotation_angle:.2f}°'
    )
    
    # Step 4: YOLO detection on rotated image
    result2 = yolo.detect_strip(rotated_image)
    
    if not result2.get('success'):
        debug.add_step(
            '04_yolo2',
            'YOLO Detection (Rotated) - FAILED',
            rotated_image,
            {'error': result2.get('error', 'Unknown error')},
            'YOLO detection failed on rotated image'
        )
        return {
            'yolo1': result1,
            'rotation_angle': rotation_angle,
            'yolo2': None
        }, rotated_image
    
    bbox2 = result2['bbox']
    conf2 = result2['confidence']
    
    # Visualize step 4
    vis2 = draw_bbox(rotated_image, bbox2, (255, 0, 0), f'YOLO 2 (conf={conf2:.2f})')
    
    debug.add_step(
        '04_yolo2',
        'YOLO Detection (Rotated)',
        vis2,
        {
            'bbox': bbox2,
            'confidence': conf2,
            'width': bbox2['x2'] - bbox2['x1'],
            'height': bbox2['y2'] - bbox2['y1'],
            'confidence_change': conf2 - conf1
        },
        f'YOLO detected strip in rotated image with confidence {conf2:.3f}'
    )
    
    # Final comparison
    # Resize images to same height for comparison
    h1, w1 = vis1.shape[:2]
    h2, w2 = vis2.shape[:2]
    max_h = max(h1, h2)
    
    w1_resized = int(w1 * max_h / h1)
    w2_resized = int(w2 * max_h / h2)
    vis1_resized = cv2.resize(vis1, (w1_resized, max_h))
    vis2_resized = cv2.resize(vis2, (w2_resized, max_h))
    
    comparison = np.hstack([vis1_resized, vis2_resized])
    cv2.putText(comparison, 'Before Rotation', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(comparison, 'After Rotation', (w1_resized + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    debug.add_step(
        '05_comparison',
        'Before/After Comparison',
        comparison,
        {
            'yolo1_confidence': conf1,
            'yolo2_confidence': conf2,
            'confidence_change': conf2 - conf1,
            'rotation_angle': rotation_angle,
            'yolo1_bbox': bbox1,
            'yolo2_bbox': bbox2
        },
        f'Comparison: Confidence changed from {conf1:.3f} to {conf2:.3f}'
    )
    
    result = {
        'yolo1': result1,
        'rotation_angle': rotation_angle,
        'yolo2': result2,
        'rotated_image': rotated_image,
        'bbox2': bbox2
    }
    
    return result, comparison


def main():
    parser = argparse.ArgumentParser(description='Test YOLO → PCA → Rotate → YOLO pipeline and save cropped images')
    parser.add_argument('images', nargs='+', help='Path(s) to test image(s)')
    parser.add_argument('--output-dir', '-o', help='Output directory for results', 
                       default='experiments/yolo_pca_pipeline')
    parser.add_argument('--crop-output', '-c', help='Output directory for cropped images (for Roboflow)', 
                       default='dataset/roboflow_upload')
    parser.add_argument('--padding', type=int, default=10, help='Padding around strip in pixels (default: 10)')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create crop output directory
    crop_output_path = Path(args.crop_output)
    crop_output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            continue
        
        image_name = Path(image_path).stem
        
        # Create debug context following existing pattern
        debug = DebugContext(
            enabled=True,
            output_dir=str(output_path),
            image_name=image_name,
            comparison_tag='yolo_pca_pipeline'
        )
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        result, final_image = process_image(image_path, debug)
        
        if result and result.get('yolo2') and result.get('rotated_image') is not None:
            # Save debug log with final comparison image
            log_path = debug.save_log(final_image)
            print(f"\nResults saved to: {log_path}")
            if log_path:
                print(f"View in browser: http://localhost:5000/review/run/{image_name}/yolo_pca_pipeline")
            
            # Crop and save the rotated strip
            rotated_image = result['rotated_image']
            bbox2 = result['bbox2']
            
            # Add padding
            padding = args.padding
            x1 = max(0, bbox2['x1'] - padding)
            y1 = max(0, bbox2['y1'] - padding)
            x2 = min(rotated_image.shape[1], bbox2['x2'] + padding)
            y2 = min(rotated_image.shape[0], bbox2['y2'] + padding)
            
            cropped_strip = rotated_image[y1:y2, x1:x2]
            
            if cropped_strip.size > 0:
                crop_filename = f"{image_name}.jpg"
                crop_path = crop_output_path / crop_filename
                cv2.imwrite(str(crop_path), cropped_strip)
                print(f"✓ Cropped image saved: {crop_path} ({cropped_strip.shape[1]}x{cropped_strip.shape[0]})")
            
            results.append(result)
        else:
            print(f"✗ Failed to process {image_path}")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} image(s)")
    print(f"Cropped images saved to: {crop_output_path}")
    print(f"{'='*60}")
    print(f"\nTo view results in browser:")
    print(f"1. Start Flask app: python app.py")
    print(f"2. Visit: http://localhost:5000/review")
    print(f"3. Or use: python scripts/serve_review.py --experiments-dir {args.output_dir}")


if __name__ == '__main__':
    main()

