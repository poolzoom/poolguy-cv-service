"""
Validation logic for strip detection results.

Provides functions to validate YOLO detections based on:
- IoU with expected bbox
- Aspect ratio
- Position (centering)
- Confidence threshold
"""

import logging
from typing import Dict, Tuple, Optional

from .context import ValidationResult

logger = logging.getLogger(__name__)


def calculate_bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bbox with 'x1', 'y1', 'x2', 'y2'
        bbox2: Second bbox with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
    x1_2, y1_2, x2_2, y2_2 = bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def check_aspect_ratio(bbox: Dict, min_ratio: float = 4.0, max_ratio: float = 15.0) -> bool:
    """
    Check if bbox has expected strip aspect ratio.
    
    Args:
        bbox: Bounding box with x1, y1, x2, y2
        min_ratio: Minimum length/width ratio
        max_ratio: Maximum length/width ratio
    
    Returns:
        True if aspect ratio is within expected range
    """
    w = bbox['x2'] - bbox['x1']
    h = bbox['y2'] - bbox['y1']
    if w <= 0 or h <= 0:
        return False
    
    ratio = max(w, h) / min(w, h)
    return min_ratio <= ratio <= max_ratio


def check_centering(
    bbox: Dict, 
    image_shape: Tuple[int, int], 
    margin: float = 0.2
) -> bool:
    """
    Check if bbox center is within the center region of the image.
    
    Args:
        bbox: Bounding box
        image_shape: (height, width) of image
        margin: Margin from edges (0.2 = 20% border)
    
    Returns:
        True if bbox center is within center region
    """
    h, w = image_shape
    center_x = (bbox['x1'] + bbox['x2']) / 2
    center_y = (bbox['y1'] + bbox['y2']) / 2
    
    return (margin * w < center_x < (1 - margin) * w and 
            margin * h < center_y < (1 - margin) * h)


def validate_yolo_result(
    bbox: Dict,
    confidence: float,
    expected_bbox: Optional[Dict] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    min_confidence: float = 0.3,
    min_iou: float = 0.2,
    check_aspect: bool = True,
    check_center: bool = False
) -> ValidationResult:
    """
    Validate a YOLO detection result.
    
    Args:
        bbox: Detected bounding box
        confidence: Detection confidence
        expected_bbox: Expected bbox location (for IoU check)
        image_shape: Image shape for centering check
        min_confidence: Minimum acceptable confidence
        min_iou: Minimum IoU with expected bbox
        check_aspect: Whether to check aspect ratio
        check_center: Whether to check centering
    
    Returns:
        ValidationResult with is_valid, reason, and checks_passed
    """
    checks = {}
    reasons = []
    
    # Confidence check
    checks['confidence'] = confidence >= min_confidence
    if not checks['confidence']:
        reasons.append(f'confidence {confidence:.2f} < {min_confidence}')
    
    # IoU check (if expected bbox provided)
    iou = 0.0
    if expected_bbox:
        iou = calculate_bbox_iou(bbox, expected_bbox)
        checks['iou'] = iou >= min_iou
        if not checks['iou']:
            reasons.append(f'IoU {iou:.2f} < {min_iou}')
    
    # Aspect ratio check
    if check_aspect:
        checks['aspect_ratio'] = check_aspect_ratio(bbox)
        if not checks['aspect_ratio']:
            reasons.append('aspect ratio outside expected range')
    
    # Centering check
    if check_center and image_shape:
        checks['centering'] = check_centering(bbox, image_shape)
        if not checks['centering']:
            reasons.append('detection not centered')
    
    is_valid = all(checks.values())
    reason = '; '.join(reasons) if reasons else 'all checks passed'
    
    return ValidationResult(
        is_valid=is_valid,
        reason=reason,
        iou=iou,
        checks_passed=checks
    )







