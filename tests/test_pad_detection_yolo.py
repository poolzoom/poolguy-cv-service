#!/usr/bin/env python3
"""
Test pad detection using the new YOLO v8 pad detection model.

Processes cropped strip images and detects pads, saving results for review.
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.utils.debug import DebugContext
from utils.image_loader import load_image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics package is not installed.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def detect_pads_yolo(image: np.ndarray, model_path: str, confidence_threshold: float = 0.25) -> dict:
    """
    Detect pads in cropped strip image using YOLO model.
    
    Args:
        image: Cropped strip image (BGR format)
        model_path: Path to YOLO model file
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary with detection results
    """
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        image,
        conf=confidence_threshold,
        verbose=False
    )
    
    detections = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # Get box coordinates (normalized 0-1)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Convert to pixel coordinates
            h, w = image.shape[:2]
            x1_px = int(x1)
            y1_px = int(y1)
            x2_px = int(x2)
            y2_px = int(y2)
            
            detections.append({
                'bbox': {
                    'x1': x1_px,
                    'y1': y1_px,
                    'x2': x2_px,
                    'y2': y2_px,
                    'width': x2_px - x1_px,
                    'height': y2_px - y1_px
                },
                'confidence': conf,
                'class': cls,
                'class_name': 'pads'
            })
    
    # Sort by y-coordinate (top to bottom)
    detections.sort(key=lambda d: d['bbox']['y1'])
    
    # Assign pad indices
    for idx, det in enumerate(detections):
        det['pad_index'] = idx
    
    return {
        'success': len(detections) > 0,
        'detections': detections,
        'count': len(detections)
    }


def process_image(image_path: str, model_path: str, debug: DebugContext, confidence_threshold: float = 0.25):
    """
    Process a single cropped strip image and detect pads.
    
    Args:
        image_path: Path to cropped strip image
        model_path: Path to YOLO pad detection model
        debug: DebugContext for logging
        confidence_threshold: Confidence threshold for detections
    
    Returns:
        Detection result dictionary
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    # Log input image
    debug.add_step(
        '00_input',
        'Input Strip Image',
        image,
        {'width': w, 'height': h, 'image_path': image_path},
        'Cropped and rotated strip image ready for pad detection'
    )
    
    # Detect pads
    result = detect_pads_yolo(image, model_path, confidence_threshold)
    
    if not result['success']:
        debug.add_step(
            'error',
            'Pad Detection Failed',
            image,
            {'error': 'No pads detected', 'confidence_threshold': confidence_threshold},
            'YOLO pad detection found no pads'
        )
        return result
    
    detections = result['detections']
    
    # Visualize detections
    vis = image.copy()
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128)   # Purple
    ]
    
    for det in detections:
        bbox = det['bbox']
        pad_idx = det['pad_index']
        conf = det['confidence']
        color = colors[pad_idx % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
        
        # Draw label
        label = f"Pad {pad_idx} ({conf:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(bbox['y1'], label_size[1] + 10)
        cv2.rectangle(vis, (bbox['x1'], label_y - label_size[1] - 5), 
                     (bbox['x1'] + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(vis, label, (bbox['x1'] + 2, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    debug.add_step(
        '01_pad_detection',
        'Pad Detection Results',
        vis,
        {
            'detection_count': len(detections),
            'confidence_threshold': confidence_threshold,
            'detections': [
                {
                    'pad_index': d['pad_index'],
                    'bbox': d['bbox'],
                    'confidence': d['confidence']
                }
                for d in detections
            ]
        },
        f'Detected {len(detections)} pads with confidence threshold {confidence_threshold}'
    )
    
    # Create summary visualization
    summary_vis = image.copy()
    cv2.putText(summary_vis, f'Detected {len(detections)} pads', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    for det in detections:
        bbox = det['bbox']
        pad_idx = det['pad_index']
        conf = det['confidence']
        color = colors[pad_idx % len(colors)]
        cv2.rectangle(summary_vis, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 3)
        cv2.putText(summary_vis, f"{pad_idx}", (bbox['x1'], bbox['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    debug.add_step(
        '02_summary',
        'Detection Summary',
        summary_vis,
        {
            'total_pads': len(detections),
            'avg_confidence': sum(d['confidence'] for d in detections) / len(detections) if detections else 0,
            'min_confidence': min(d['confidence'] for d in detections) if detections else 0,
            'max_confidence': max(d['confidence'] for d in detections) if detections else 0
        },
        f'Summary: {len(detections)} pads detected'
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test pad detection with YOLO v8 model')
    parser.add_argument('images', nargs='+', help='Path(s) to cropped strip image(s)')
    parser.add_argument('--model', '-m', default='models/pad_detection/best.pt',
                       help='Path to YOLO pad detection model (default: models/pad_detection/best.pt)')
    parser.add_argument('--output-dir', '-o', default='experiments/pad_detection',
                       help='Output directory for results (default: experiments/pad_detection)')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    for image_path in args.images:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        image_name = Path(image_path).stem
        
        # Create debug context
        debug = DebugContext(
            enabled=True,
            output_dir=str(output_path),
            image_name=image_name,
            comparison_tag='pad_detection_yolo'
        )
        
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")
        
        result = process_image(image_path, args.model, debug, args.confidence)
        
        if result:
            # Save debug log
            final_vis = load_image(image_path)
            if final_vis is not None:
                # Draw detections on final image
                for det in result.get('detections', []):
                    bbox = det['bbox']
                    pad_idx = det['pad_index']
                    conf = det['confidence']
                    color = (0, 255, 0) if conf > 0.5 else (0, 165, 255)
                    cv2.rectangle(final_vis, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
                    cv2.putText(final_vis, f"P{pad_idx}", (bbox['x1'], bbox['y1'] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            log_path = debug.save_log(final_vis)
            print(f"\n✓ Results saved to: {log_path}")
            if log_path:
                print(f"  View in browser: http://localhost:5000/review/run/{image_name}/pad_detection_yolo")
            print(f"  Detected {result.get('count', 0)} pads")
            
            results.append({
                'image': image_name,
                'result': result,
                'log_path': log_path
            })
        else:
            print(f"✗ Failed to process {image_path}")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} image(s)")
    print(f"{'='*60}")
    print(f"\nTo view results in browser:")
    print(f"1. Start Flask app: python app.py")
    print(f"2. Visit: http://localhost:5000/review")
    print(f"3. Or use: python scripts/serve_review.py --experiments-dir {args.output_dir}")


if __name__ == '__main__':
    main()

