"""
Test the full pipeline: YOLO strip detection → Pad detection → Color extraction
"""

import cv2
import numpy as np
import logging
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.detection.yolo_detector import YoloDetector
from services.pipeline.steps.strip_detection import StripDetectionService
from services.pipeline.steps.pad_detection import PadDetectionService
from services.pipeline.steps.color_extraction import ColorExtractionService
from utils.image_loader import load_image
from utils.coordinate_transform import transform_pad_coordinates_to_absolute, crop_image_to_strip
from utils.visual_logger import VisualLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_colors_from_strip_region(
    color_service: ColorExtractionService,
    pad_detection_service: PadDetectionService,
    image: np.ndarray,
    strip_region: dict,
    expected_pad_count: int = 6
) -> dict:
    """
    Extract pad colors from a pre-detected strip region using new pipeline.
    
    Uses: StripDetection → PadDetection → ColorExtraction (all in memory)
    """
    from services.interfaces import StripRegion
    
    # Convert strip_region to StripRegion format
    strip_region_obj: StripRegion = {
        'left': strip_region['left'],
        'top': strip_region['top'],
        'right': strip_region['right'],
        'bottom': strip_region['bottom'],
        'width': strip_region.get('width', strip_region['right'] - strip_region['left']),
        'height': strip_region.get('height', strip_region['bottom'] - strip_region['top']),
        'confidence': strip_region.get('confidence', 0.0),
        'detection_method': strip_region.get('detection_method', 'unknown'),
        'orientation': strip_region.get('orientation', 'vertical'),
        'angle': strip_region.get('angle', 0.0)
    }
    
    # Crop image to strip (in memory)
    strip_image, crop_error = crop_image_to_strip(image, strip_region_obj)
    if strip_image is None:
        return {
            'success': False,
            'error': crop_error or 'Cropped strip region is empty'
        }
    
    # Detect pads in cropped strip
    pad_detection_result = pad_detection_service.detect_pads_in_strip(
        strip_image=strip_image,
        strip_region=strip_region_obj,
        expected_pad_count=expected_pad_count
    )
    
    if not pad_detection_result['success']:
        return {
            'success': False,
            'error': pad_detection_result.get('error', 'Pad detection failed'),
            'error_code': pad_detection_result.get('error_code', 'PAD_DETECTION_FAILED')
        }
    
    # Transform pad coordinates to absolute
    relative_pads = pad_detection_result['pads']
    absolute_pads = transform_pad_coordinates_to_absolute(relative_pads, strip_region_obj)
    
    # Extract colors using new pipeline (in memory, no temp files)
    result = color_service.extract_colors(
        image=strip_image,
        pad_regions=relative_pads,  # Use relative pads since we're passing strip_image
        expected_pad_count=expected_pad_count,
        normalize_white=True
    )
    
    # Update pad regions with absolute coordinates
    if result.get('success') and 'data' in result:
        pads = result['data'].get('pads', [])
        for i, pad in enumerate(pads):
            if i < len(absolute_pads):
                pad['region'] = absolute_pads[i]
    
    return result


def test_full_pipeline(image_path: str, expected_pad_count: int = 6):
    """Test the full pipeline: YOLO → Pad Detection → Color Extraction"""
    print(f"\n{'='*70}")
    print(f"FULL PIPELINE TEST: {Path(image_path).name}")
    print(f"{'='*70}")
    
    image_name = Path(image_path).stem
    log_output_dir = f"tests/fixtures/detection_logs/{image_name}"
    
    # Step 1: Load image
    print("\n[Step 1] Loading image...")
    image = load_image(image_path)
    if image is None:
        print("✗ Failed to load image")
        return None
    
    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h} pixels")
    
    # Step 2: YOLO Strip Detection
    print("\n[Step 2] YOLO Strip Detection...")
    yolo_detector = YoloDetector()
    strip_detection_service = StripDetectionService(
        enable_visual_logging=True,
        log_output_dir=log_output_dir,
        yolo_detector=yolo_detector
    )
    
    strip_result = strip_detection_service.detect_strip(
        image=image,
        image_name=image_name,
        expected_pad_count=expected_pad_count
    )
    
    if not strip_result.get('success'):
        print(f"✗ Strip detection failed: {strip_result.get('error')}")
        return None
    
    strip_region = strip_result['strip_region']
    yolo_confidence = strip_result.get('yolo_confidence', 0.0)
    print(f"✓ Strip detected!")
    print(f"  Region: ({strip_region['left']}, {strip_region['top']}) to ({strip_region['right']}, {strip_region['bottom']})")
    print(f"  Size: {strip_region['right'] - strip_region['left']}x{strip_region['bottom'] - strip_region['top']}")
    print(f"  YOLO Confidence: {yolo_confidence:.3f}")
    
    # Step 3: Pad Detection & Color Extraction
    print("\n[Step 3] Pad Detection & Color Extraction...")
    color_service = ColorExtractionService()
    pad_detection_service = PadDetectionService()
    
    color_result = extract_colors_from_strip_region(
        color_service=color_service,
        pad_detection_service=pad_detection_service,
        image=image,
        strip_region=strip_region,
        expected_pad_count=expected_pad_count
    )
    
    if not color_result.get('success'):
        print(f"✗ Color extraction failed: {color_result.get('error')}")
        print(f"  Detected {color_result.get('detected_count', 0)} pads")
        return {
            'strip_detection': strip_result,
            'color_extraction': color_result,
            'success': False
        }
    
    pads = color_result.get('data', {}).get('pads', [])
    overall_confidence = color_result.get('data', {}).get('overall_confidence', 0.0)
    
    print(f"✓ Color extraction succeeded!")
    print(f"  Detected {len(pads)} pads")
    print(f"  Overall confidence: {overall_confidence:.3f}")
    
    # Display pad information
    print("\n  Pad Details:")
    for i, pad in enumerate(pads):
        lab = pad.get('lab', {})
        region = pad.get('region', {})
        confidence = pad.get('confidence', 0.0)
        print(f"    Pad {i+1}: L={lab.get('L', 0):.1f}, a={lab.get('a', 0):.1f}, b={lab.get('b', 0):.1f}, conf={confidence:.3f}")
        if region:
            print(f"             Region: ({region.get('x', 0)}, {region.get('y', 0)}) size {region.get('width', 0)}x{region.get('height', 0)}")
    
    # Step 4: Create visualization
    print("\n[Step 4] Creating visualization...")
    vis_image = image.copy()
    
    # Draw strip region (green)
    cv2.rectangle(vis_image,
                 (strip_region['left'], strip_region['top']),
                 (strip_region['right'], strip_region['bottom']),
                 (0, 255, 0), 3)
    cv2.putText(vis_image, f'Strip (YOLO: {yolo_confidence:.2f})',
               (strip_region['left'], strip_region['top'] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Draw detected pads (blue)
    for i, pad in enumerate(pads):
        region = pad.get('region', {})
        if region:
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('width', 0)
            h = region.get('height', 0)
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis_image, f'P{i+1}',
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save visualization
    vis_path = Path(log_output_dir) / f"{image_name}_full_pipeline.jpg"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), vis_image)
    print(f"✓ Visualization saved to: {vis_path}")
    
    # Save summary JSON
    summary = {
        'image_path': image_path,
        'image_name': image_name,
        'strip_detection': {
            'success': True,
            'strip_region': strip_region,
            'yolo_confidence': yolo_confidence,
            'method': strip_result.get('method', 'unknown')
        },
        'color_extraction': {
            'success': True,
            'pads_detected': len(pads),
            'overall_confidence': overall_confidence,
            'pads': pads
        },
        'full_pipeline_success': True
    }
    
    summary_path = Path(log_output_dir) / f"{image_name}_full_pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")
    
    return summary


if __name__ == '__main__':
    # Test images
    test_images = [
        ("tests/fixtures/PXL_20250427_161114135.jpg", 6),
        ("tests/fixtures/PXL_20250427_161135607.MP.jpg", 6),
        ("tests/fixtures/PXL_20250428_134210676.MP.jpg", 6),
        ("tests/fixtures/PXL_20251116_223654116.jpg", 6),
    ]
    
    results = []
    for img_path, pad_count in test_images:
        if os.path.exists(img_path):
            result = test_full_pipeline(img_path, expected_pad_count=pad_count)
            if result:
                results.append(result)
        else:
            print(f"\n⚠ Image not found: {img_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r.get('full_pipeline_success', False))
    print(f"Successful pipelines: {successful}/{len(results)}")
    
    if results:
        for r in results:
            img_name = r.get('image_name', 'unknown')
            success = r.get('full_pipeline_success', False)
            pads = r.get('color_extraction', {}).get('pads_detected', 0)
            conf = r.get('color_extraction', {}).get('overall_confidence', 0.0)
            status = "✓" if success else "✗"
            print(f"  {status} {img_name}: {pads} pads, conf={conf:.3f}")

