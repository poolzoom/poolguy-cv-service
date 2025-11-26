#!/usr/bin/env python3
"""
Quick test script for refinement system.

Tests refinement on a single image to verify the system works.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np
from datetime import datetime
from services.detection.yolo_detector import YoloDetector
from services.refinement import StripRefiner
from services.utils.debug import DebugContext
from config.refinement_config import get_config
from utils.image_loader import load_image


def test_refinement(image_path: str):
    """Test refinement on a single image."""
    print(f"\n{'='*60}")
    print(f"Testing Refinement System")
    print(f"Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Load image
    print("1. Loading image...")
    image = load_image(image_path)
    if image is None:
        print(f"   ERROR: Failed to load image")
        return
    print(f"   ✓ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # YOLO detection
    print("\n2. Running YOLO detection...")
    yolo_detector = YoloDetector()
    if not yolo_detector.is_available():
        print("   ERROR: YOLO detector not available")
        return
    
    yolo_result = yolo_detector.detect_strip(image)
    if not yolo_result.get('success'):
        print(f"   ERROR: YOLO detection failed: {yolo_result.get('error')}")
        return
    
    yolo_bbox = yolo_result['bbox']
    print(f"   ✓ YOLO detected: ({yolo_bbox['x1']}, {yolo_bbox['y1']}) -> ({yolo_bbox['x2']}, {yolo_bbox['y2']})")
    print(f"   ✓ Confidence: {yolo_result.get('confidence', 0):.3f}")
    
    # Setup refinement
    print("\n3. Setting up refinement...")
    config = get_config('balanced')
    refiner = StripRefiner(config)
    print("   ✓ Refiner initialized")
    
    # Setup debug with timestamped directory to avoid overwriting
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{image_name}_{timestamp}"
    debug = DebugContext(
        enabled=True,
        output_dir=f"experiments/refinement/test_single/{image_name}",
        image_name=run_name
    )
    # Set strategy name after initialization
    if debug.visual_logger:
        debug.visual_logger.strategy_name = "pipeline"
    print(f"   ✓ Debug context created (run: {run_name})")
    
    # Run refinement
    print("\n4. Running refinement pipeline...")
    try:
        refined_strip = refiner.refine(image, yolo_bbox, debug)
        print("   ✓ Refinement complete")
        
        # Save debug log
        vis = image.copy()
        cv2.rectangle(vis, (yolo_bbox['x1'], yolo_bbox['y1']),
                     (yolo_bbox['x2'], yolo_bbox['y2']), (255, 0, 0), 3)  # YOLO in blue
        cv2.rectangle(vis, (refined_strip['left'], refined_strip['top']),
                     (refined_strip['right'], refined_strip['bottom']), (0, 255, 0), 3)  # Refined in green
        
        log_path = debug.save_log(vis)
        if log_path:
            print(f"   ✓ Debug log saved: {log_path}")
        else:
            # Fallback: construct path from debug context
            debug_dir = f"experiments/refinement/test_single/{image_name}/{run_name}/pipeline"
            if os.path.exists(debug_dir):
                print(f"   ✓ Debug log saved: {debug_dir}")
                log_path = debug_dir
            else:
                print(f"   ⚠ Debug log path not available")
        
        # Print results
        print("\n5. Results:")
        print(f"   YOLO bbox:     ({yolo_bbox['x1']}, {yolo_bbox['y1']}) -> ({yolo_bbox['x2']}, {yolo_bbox['y2']})")
        print(f"   Refined bbox:  ({refined_strip['left']}, {refined_strip['top']}) -> ({refined_strip['right']}, {refined_strip['bottom']})")
        
        yolo_area = (yolo_bbox['x2'] - yolo_bbox['x1']) * (yolo_bbox['y2'] - yolo_bbox['y1'])
        refined_area = refined_strip['width'] * refined_strip['height']
        reduction = (1.0 - refined_area / yolo_area) * 100 if yolo_area > 0 else 0
        print(f"   Area reduction: {reduction:.1f}%")
        print(f"   Orientation:   {refined_strip['orientation']}")
        print(f"   Angle:          {refined_strip['angle']:.1f}°")
        
        print(f"\n{'='*60}")
        print("✓ Test complete! Check debug images in:")
        print(f"  {log_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"   ERROR: Refinement failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_refinement.py <image_path>")
        print("Example: python scripts/test_refinement.py tests/fixtures/PXL_20250427_161114135.jpg")
        sys.exit(1)
    
    test_refinement(sys.argv[1])

