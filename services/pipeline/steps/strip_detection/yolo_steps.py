"""
YOLO detection steps for strip detection pipeline.

Provides step functions that execute YOLO detection with various strategies:
- Direct detection
- Detection with center crop
- Detection with expected area crop
- Detection with confidence sweep
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from services.detection.yolo_detector import YoloDetector
from services.utils.debug import DebugContext
from .context import YoloResult, DetectionContext
from .validation import validate_yolo_result, calculate_bbox_iou, check_centering

logger = logging.getLogger(__name__)


def run_yolo_detection(
    yolo_detector: YoloDetector,
    image: np.ndarray,
    step_name: str = "yolo"
) -> YoloResult:
    """
    Run basic YOLO detection on an image.
    
    Args:
        yolo_detector: YOLO detector instance
        image: Image to detect on
        step_name: Name for logging/debugging
    
    Returns:
        YoloResult with detection outcome
    """
    result = yolo_detector.detect_strip(image)
    
    return YoloResult(
        success=result.get('success', False),
        bbox=result.get('bbox'),
        confidence=result.get('confidence', 0.0),
        error=result.get('error'),
        step_name=step_name
    )


def run_yolo_with_crop(
    yolo_detector: YoloDetector,
    image: np.ndarray,
    crop_region: Dict,
    step_name: str = "yolo_cropped"
) -> Tuple[YoloResult, Tuple[int, int]]:
    """
    Run YOLO detection on a cropped region.
    
    Args:
        yolo_detector: YOLO detector instance
        image: Full image
        crop_region: Dict with x1, y1, x2, y2
        step_name: Name for logging
    
    Returns:
        Tuple of (YoloResult with adjusted coords, crop_offset)
    """
    x1, y1 = crop_region['x1'], crop_region['y1']
    x2, y2 = crop_region['x2'], crop_region['y2']
    
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return YoloResult(success=False, error="Empty crop", step_name=step_name), (0, 0)
    
    result = run_yolo_detection(yolo_detector, cropped, step_name)
    
    # Adjust coordinates back to full image space
    if result.success and result.bbox:
        result.bbox = {
            'x1': result.bbox['x1'] + x1,
            'y1': result.bbox['y1'] + y1,
            'x2': result.bbox['x2'] + x1,
            'y2': result.bbox['y2'] + y1
        }
    
    return result, (x1, y1)


def run_yolo_with_center_crop(
    yolo_detector: YoloDetector,
    image: np.ndarray,
    margin: float = 0.2,
    step_name: str = "yolo_centered"
) -> Tuple[YoloResult, Tuple[int, int]]:
    """
    Run YOLO on center-cropped image (removes edges).
    
    Args:
        yolo_detector: YOLO detector instance
        image: Full image
        margin: Percentage to crop from each edge (0.2 = 20%)
        step_name: Name for logging
    
    Returns:
        Tuple of (YoloResult with adjusted coords, crop_offset)
    """
    h, w = image.shape[:2]
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    crop_region = {
        'x1': margin_x,
        'y1': margin_y,
        'x2': w - margin_x,
        'y2': h - margin_y
    }
    
    return run_yolo_with_crop(yolo_detector, image, crop_region, step_name)


def run_yolo_with_expected_area(
    yolo_detector: YoloDetector,
    image: np.ndarray,
    expected_bbox: Dict,
    expand_margin: float = 0.3,
    step_name: str = "yolo_expected"
) -> Tuple[YoloResult, Tuple[int, int]]:
    """
    Run YOLO on area around expected bbox location.
    
    Args:
        yolo_detector: YOLO detector instance
        image: Full image
        expected_bbox: Expected bbox location
        expand_margin: How much to expand expected area
        step_name: Name for logging
    
    Returns:
        Tuple of (YoloResult with adjusted coords, crop_offset)
    """
    h, w = image.shape[:2]
    
    exp_w = (expected_bbox['x2'] - expected_bbox['x1']) * expand_margin
    exp_h = (expected_bbox['y2'] - expected_bbox['y1']) * expand_margin
    
    crop_region = {
        'x1': max(0, int(expected_bbox['x1'] - exp_w)),
        'y1': max(0, int(expected_bbox['y1'] - exp_h)),
        'x2': min(w, int(expected_bbox['x2'] + exp_w)),
        'y2': min(h, int(expected_bbox['y2'] + exp_h))
    }
    
    return run_yolo_with_crop(yolo_detector, image, crop_region, step_name)


def run_yolo_with_fallbacks(
    yolo_detector: YoloDetector,
    image: np.ndarray,
    expected_bbox: Optional[Dict] = None,
    config: Optional[Dict] = None,
    debug: Optional[DebugContext] = None,
    step_prefix: str = "01_05"
) -> YoloResult:
    """
    Run YOLO with multiple fallback strategies until valid result.
    
    For rotated images, we ALWAYS crop to expected area first to avoid
    the white rotation padding confusing YOLO.
    
    Strategies tried in order:
    1. Crop to expected area (if provided) - avoids rotation padding
    2. Center crop (20% margin) - focuses on center where strip likely is
    3. Use expected_bbox directly as fallback
    
    Args:
        yolo_detector: YOLO detector instance
        image: Image to detect on
        expected_bbox: Expected location (from previous detection)
        config: Configuration dict
        debug: Debug context for visualization
        step_prefix: Prefix for debug step names
    
    Returns:
        Best YoloResult from strategies
    """
    cfg = config or {}
    min_iou = cfg.get('yolo2_min_iou', 0.3)  # Higher threshold for cropped detection
    
    img_shape = image.shape[:2]
    
    # Strategy 1: Always crop to expected area first (avoids rotation padding)
    if expected_bbox:
        # Crop to expected area with generous margin
        result1, offset = run_yolo_with_expected_area(
            yolo_detector, image, expected_bbox, 
            expand_margin=0.4,  # 40% expansion for safety
            step_name=f"{step_prefix}_cropped"
        )
        
        if result1.success:
            # Validate: check IoU with expected bbox
            iou = calculate_bbox_iou(result1.bbox, expected_bbox)
            is_valid = iou >= min_iou
            
            if debug:
                _visualize_cropped_detection(debug, image, expected_bbox, result1, iou,
                                            f"{step_prefix}_yolo2", is_valid)
            
            if is_valid:
                logger.info(f'YOLO 2 found strip in expected area (IoU={iou:.3f})')
                return result1
            else:
                logger.warning(f'YOLO 2 detection IoU ({iou:.3f}) < threshold ({min_iou})')
        else:
            if debug:
                _visualize_cropped_detection(debug, image, expected_bbox, result1, 0.0,
                                            f"{step_prefix}_yolo2", False)
            logger.warning('YOLO 2 failed to detect in expected area')
    
    # Strategy 2: Center crop - only look in center of image
    result2, offset = run_yolo_with_center_crop(
        yolo_detector, image, margin=0.2, step_name=f"{step_prefix}_centered"
    )
    
    if result2.success:
        # Check if detection is near expected area
        if expected_bbox:
            iou = calculate_bbox_iou(result2.bbox, expected_bbox)
            is_valid = iou >= min_iou * 0.5  # Lower threshold for center crop
            if debug:
                _visualize_center_crop(debug, image, result2, f"{step_prefix}a_centered",
                                      expected_bbox, iou)
            if is_valid:
                logger.info(f'YOLO 2 found strip in center crop (IoU={iou:.3f})')
                return result2
        else:
            if debug:
                _visualize_center_crop(debug, image, result2, f"{step_prefix}a_centered")
            return result2
    
    # Strategy 3: Use expected bbox directly as fallback
    if expected_bbox:
        logger.warning('All YOLO 2 strategies failed, using expected bbox from YOLO 1')
        if debug:
            _visualize_fallback(debug, image, expected_bbox, f"{step_prefix}b_fallback")
        return YoloResult(
            success=True,
            bbox=expected_bbox,
            confidence=0.5,  # Low confidence for fallback
            step_name=f"{step_prefix}_fallback"
        )
    
    return YoloResult(success=False, error="All strategies failed", step_name=step_prefix)


def _visualize_cropped_detection(
    debug: DebugContext,
    image: np.ndarray,
    expected_bbox: Dict,
    result: YoloResult,
    iou: float,
    step_id: str,
    is_valid: bool
):
    """Helper to visualize detection in cropped expected area."""
    vis = image.copy()
    h, w = image.shape[:2]
    
    # Show the crop region
    exp_w = (expected_bbox['x2'] - expected_bbox['x1']) * 0.4
    exp_h = (expected_bbox['y2'] - expected_bbox['y1']) * 0.4
    crop_x1 = max(0, int(expected_bbox['x1'] - exp_w))
    crop_y1 = max(0, int(expected_bbox['y1'] - exp_h))
    crop_x2 = min(w, int(expected_bbox['x2'] + exp_w))
    crop_y2 = min(h, int(expected_bbox['y2'] + exp_h))
    
    # Draw crop region in yellow
    cv2.rectangle(vis, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)
    cv2.putText(vis, 'Crop region', (crop_x1, crop_y1 - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw expected bbox in cyan
    cv2.rectangle(vis, (expected_bbox['x1'], expected_bbox['y1']),
                 (expected_bbox['x2'], expected_bbox['y2']), (255, 255, 0), 2)
    cv2.putText(vis, 'Expected', (expected_bbox['x1'], expected_bbox['y1'] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw detection in green (valid) or red (invalid)
    if result.bbox:
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.rectangle(vis, (result.bbox['x1'], result.bbox['y1']),
                     (result.bbox['x2'], result.bbox['y2']), color, 3)
        status = "VALID" if is_valid else "REJECTED"
        cv2.putText(vis, f'YOLO 2 ({status}, IoU={iou:.2f})', 
                   (result.bbox['x1'], result.bbox['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(vis, 'YOLO 2: No detection', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    debug.add_step(step_id, 'YOLO 2 Detection (Cropped)', vis, {
        'bbox': result.bbox,
        'confidence': result.confidence,
        'iou': iou,
        'is_valid': is_valid,
        'expected_bbox': expected_bbox,
        'crop_region': {'x1': crop_x1, 'y1': crop_y1, 'x2': crop_x2, 'y2': crop_y2}
    }, f'YOLO 2 in cropped area: {"VALID" if is_valid else "REJECTED"} (IoU={iou:.3f})')


def _visualize_center_crop(
    debug: DebugContext,
    image: np.ndarray,
    result: YoloResult,
    step_id: str,
    expected_bbox: Optional[Dict] = None,
    iou: float = 0.0
):
    """Helper to visualize center crop detection."""
    vis = image.copy()
    h, w = image.shape[:2]
    margin_x, margin_y = int(w * 0.2), int(h * 0.2)
    
    # Draw center crop region
    cv2.rectangle(vis, (margin_x, margin_y), (w - margin_x, h - margin_y), 
                 (255, 255, 0), 2)
    cv2.putText(vis, 'Center crop region', (margin_x, margin_y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if expected_bbox:
        cv2.rectangle(vis, (expected_bbox['x1'], expected_bbox['y1']),
                     (expected_bbox['x2'], expected_bbox['y2']), (0, 255, 255), 2)
    
    if result.bbox:
        cv2.rectangle(vis, (result.bbox['x1'], result.bbox['y1']),
                     (result.bbox['x2'], result.bbox['y2']), (0, 255, 0), 3)
        label = f'YOLO 2 (conf={result.confidence:.2f}'
        if expected_bbox:
            label += f', IoU={iou:.2f}'
        label += ')'
        cv2.putText(vis, label, (result.bbox['x1'], result.bbox['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    debug.add_step(step_id, 'Center Crop Detection', vis, {
        'result': result.to_dict(),
        'iou': iou if expected_bbox else None
    }, f'YOLO 2 center crop: conf={result.confidence:.3f}' + 
       (f', IoU={iou:.3f}' if expected_bbox else ''))


def _visualize_fallback(
    debug: DebugContext,
    image: np.ndarray,
    expected_bbox: Dict,
    step_id: str
):
    """Helper to visualize fallback to expected bbox."""
    vis = image.copy()
    
    cv2.rectangle(vis, (expected_bbox['x1'], expected_bbox['y1']),
                 (expected_bbox['x2'], expected_bbox['y2']), (255, 165, 0), 3)
    cv2.putText(vis, 'FALLBACK: Using expected bbox from YOLO 1', 
               (expected_bbox['x1'], expected_bbox['y1'] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    debug.add_step(step_id, 'Fallback to Expected', vis, {
        'bbox': expected_bbox,
        'confidence': 0.5,
        'reason': 'All YOLO 2 strategies failed'
    }, 'Using expected bbox from YOLO 1 (transformed to rotated space)')

