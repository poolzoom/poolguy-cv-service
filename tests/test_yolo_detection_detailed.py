"""
Detailed YOLO detection test with lower confidence threshold.
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ultralytics import YOLO
from config.yolo_config import MODEL_PATH
from utils.image_loader import load_image

def test_yolo_detailed(image_path: str):
    """Test YOLO with detailed output and lower confidence."""
    print(f"Testing YOLO on: {image_path}")
    print("=" * 60)
    
    # Load image
    image = load_image(image_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load model
    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    
    # Test with different confidence thresholds
    for conf_threshold in [0.1, 0.05, 0.01, 0.001]:
        print(f"\n--- Testing with confidence threshold: {conf_threshold} ---")
        results = model.predict(
            image,
            imgsz=640,
            conf=conf_threshold,
            verbose=False
        )
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            print(f"Found {len(boxes)} detection(s):")
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)
                print(f"  Detection {i+1}:")
                print(f"    Confidence: {conf:.4f}")
                print(f"    BBox: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"    Size: {x2-x1}x{y2-y1}")
                
                # Visualize
                vis = image.copy()
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(vis, f'Conf: {conf:.3f}',
                           (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                output = f'yolo_detection_conf_{conf_threshold}.jpg'
                cv2.imwrite(output, vis)
                print(f"    Saved to: {output}")
        else:
            print(f"  No detections found with confidence >= {conf_threshold}")

if __name__ == '__main__':
    test_yolo_detailed('tests/fixtures/PXL_20250427_161114135.jpg')



