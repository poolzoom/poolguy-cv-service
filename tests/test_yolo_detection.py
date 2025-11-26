"""
Test YOLO detection on a single image with comprehensive visual logging.
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.detection.yolo_detector import YoloDetector
from services.pipeline.steps.strip_detection import StripDetectionService
from utils.image_loader import load_image
from utils.visual_logger import VisualLogger

def test_yolo_detection(image_path: str, enable_logging: bool = True):
    """
    Test YOLO detection on a single image with comprehensive logging.
    
    Args:
        image_path: Path to test image
        enable_logging: Whether to enable visual logging
    """
    print(f"Testing YOLO detection on: {image_path}")
    print("=" * 60)
    
    # Get image name for logging
    image_name = Path(image_path).stem
    
    # Initialize visual logger
    visual_logger = None
    if enable_logging:
        visual_logger = VisualLogger()
        visual_logger.start_log('yolo_detection', image_name)
    
    # Load image
    try:
        image = load_image(image_path)
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        if visual_logger:
            visual_logger.add_step('00_input', 'Input image', image.copy(), {
                'width': int(image.shape[1]),
                'height': int(image.shape[0]),
                'channels': int(image.shape[2])
            })
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return
    
    # Initialize YOLO detector
    try:
        yolo_detector = YoloDetector()
        if not yolo_detector.is_available():
            print("✗ YOLO model not available")
            if visual_logger:
                vis_error = image.copy()
                cv2.putText(vis_error, 'YOLO model not available',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                visual_logger.add_step('99_error', 'YOLO model not available', vis_error)
                visual_logger.save_log(vis_error)
            return
        print("✓ YOLO detector initialized")
        
        if visual_logger:
            vis_init = image.copy()
            cv2.putText(vis_init, f'YOLO Model: {yolo_detector.model_path}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_init, f'Confidence Threshold: {yolo_detector.confidence_threshold}',
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            visual_logger.add_step('00_yolo_init', 'YOLO detector initialized', vis_init, {
                'model_path': yolo_detector.model_path,
                'confidence_threshold': yolo_detector.confidence_threshold,
                'img_size': yolo_detector.img_size
            })
    except Exception as e:
        print(f"✗ Failed to initialize YOLO detector: {e}")
        if visual_logger:
            vis_error = image.copy()
            cv2.putText(vis_error, f'YOLO init failed: {str(e)}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            visual_logger.add_step('99_error', 'YOLO initialization failed', vis_error)
            visual_logger.save_log(vis_error)
        return
    
    # Run YOLO detection
    print("\n--- YOLO Detection ---")
    yolo_result = yolo_detector.detect_strip(image)
    
    if yolo_result.get('success'):
        bbox = yolo_result['bbox']
        confidence = yolo_result['confidence']
        print(f"✓ Strip detected!")
        print(f"  Bounding box: ({bbox['x1']}, {bbox['y1']}) to ({bbox['x2']}, {bbox['y2']})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Size: {bbox['x2'] - bbox['x1']}x{bbox['y2'] - bbox['y1']} pixels")
        
        # Visualize YOLO detection
        vis_yolo = image.copy()
        cv2.rectangle(vis_yolo, 
                     (bbox['x1'], bbox['y1']), 
                     (bbox['x2'], bbox['y2']), 
                     (0, 255, 0), 3)
        cv2.putText(vis_yolo, f'YOLO Detection: {confidence:.3f}',
                   (bbox['x1'], bbox['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(vis_yolo, f'Size: {bbox["x2"]-bbox["x1"]}x{bbox["y2"]-bbox["y1"]}',
                   (bbox['x1'], bbox['y2'] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if visual_logger:
            visual_logger.add_step('01_yolo_detection', 'YOLO strip detection', vis_yolo, {
                'bbox': bbox,
                'confidence': confidence,
                'width': bbox['x2'] - bbox['x1'],
                'height': bbox['y2'] - bbox['y1']
            })
        
        # Crop to detected region
        cropped = image[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]
        if visual_logger and cropped.size > 0:
            visual_logger.add_step('02_cropped_region', 'Cropped to YOLO-detected region', cropped, {
                'crop_size': f"{cropped.shape[1]}x{cropped.shape[0]}"
            })
        
        # Test full strip detection pipeline with logging
        print("\n--- Full Strip Detection Pipeline (YOLO + OpenCV) ---")
        strip_service = StripDetectionService(
            enable_visual_logging=enable_logging,
            log_output_dir='tests/fixtures/detection_logs' if enable_logging else None
        )
        strip_result = strip_service.detect_strip(image, image_name=image_name)
        
        if strip_result.get('success'):
            print("✓ Full pipeline succeeded!")
            print(f"  Method: {strip_result.get('method', 'unknown')}")
            print(f"  Confidence: {strip_result.get('confidence', 0):.3f}")
            strip_region = strip_result.get('strip_region', {})
            print(f"  Strip region: {strip_region}")
            
            # Create final visualization
            vis_final = image.copy()
            
            # Draw YOLO bbox in green
            if yolo_result.get('success'):
                yolo_bbox = yolo_result['bbox']
                cv2.rectangle(vis_final,
                            (yolo_bbox['x1'], yolo_bbox['y1']),
                            (yolo_bbox['x2'], yolo_bbox['y2']),
                            (0, 255, 0), 2)
                cv2.putText(vis_final, f'YOLO: {yolo_result["confidence"]:.3f}',
                           (yolo_bbox['x1'], yolo_bbox['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw final strip region in blue
            if 'strip_region' in strip_result:
                sr = strip_result['strip_region']
                cv2.rectangle(vis_final,
                            (sr['left'], sr['top']),
                            (sr['right'], sr['bottom']),
                            (255, 0, 0), 3)
                cv2.putText(vis_final, f'Final: {strip_result.get("method", "unknown")}',
                           (sr['left'], sr['top'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            
            if visual_logger:
                visual_logger.add_step('03_final_result', 'Final detection result', vis_final, {
                    'method': strip_result.get('method'),
                    'confidence': strip_result.get('confidence'),
                    'strip_region': strip_result.get('strip_region'),
                    'yolo_bbox': yolo_result.get('bbox'),
                    'yolo_confidence': yolo_result.get('confidence')
                })
            
            # Save log
            if visual_logger:
                log_path = visual_logger.save_log(vis_final)
                print(f"\n✓ Visual log saved to: {log_path}")
                
                # Also save a summary JSON
                summary = {
                    'image_path': image_path,
                    'image_name': image_name,
                    'yolo_result': yolo_result,
                    'strip_result': strip_result,
                    'success': True
                }
                summary_path = Path(log_path) / 'summary.json'
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"✓ Summary JSON saved to: {summary_path}")
        else:
            print(f"✗ Full pipeline failed: {strip_result.get('error', 'Unknown error')}")
            if visual_logger:
                vis_error = image.copy()
                cv2.putText(vis_error, f'Pipeline failed: {strip_result.get("error", "Unknown")}',
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                visual_logger.add_step('99_pipeline_failed', 'Pipeline failed', vis_error, {
                    'error': strip_result.get('error')
                })
                visual_logger.save_log(vis_error)
    else:
        error_msg = yolo_result.get('error', 'No strip detected')
        print(f"✗ YOLO detection failed: {error_msg}")
        
        if visual_logger:
            vis_error = image.copy()
            cv2.putText(vis_error, f'YOLO Failed: {error_msg}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            visual_logger.add_step('99_yolo_failed', 'YOLO detection failed', vis_error, {
                'error': error_msg
            })
            visual_logger.save_log(vis_error)

if __name__ == '__main__':
    image_path = 'tests/fixtures/PXL_20250427_161114135.jpg'
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    test_yolo_detection(image_path, enable_logging=True)

