"""
Adapter functions to convert detector outputs to standardized StripRegion format.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.interfaces import StripRegion


def yolo_result_to_strip_region(
    yolo_result: Dict,
    image_shape: Tuple[int, int]
) -> Optional[StripRegion]:
    """
    Convert YOLO detector result to StripRegion format.
    
    Args:
        yolo_result: Result from YoloDetector.detect_strip()
        image_shape: (height, width) of original image
        
    Returns:
        StripRegion if successful, None otherwise
    """
    if not yolo_result.get('success'):
        return None
    
    bbox = yolo_result.get('bbox', {})
    if not bbox:
        return None
    
    x1 = bbox.get('x1', 0)
    y1 = bbox.get('y1', 0)
    x2 = bbox.get('x2', 0)
    y2 = bbox.get('y2', 0)
    
    # Calculate dimensions
    width = x2 - x1
    height = y2 - y1
    
    # Determine orientation
    orientation = 'vertical' if height > width else 'horizontal'
    
    return StripRegion(
        left=x1,
        top=y1,
        right=x2,
        bottom=y2,
        width=width,
        height=height,
        confidence=yolo_result.get('confidence', 0.0),
        detection_method='yolo',
        orientation=orientation,
        angle=0.0  # YOLO doesn't provide angle
    )


def opencv_result_to_strip_region(
    strip_region_dict: Optional[Dict],
    orientation: str,
    angle: float,
    method: str = 'opencv',
    confidence: float = 0.5
) -> Optional[StripRegion]:
    """
    Convert OpenCV detector result to StripRegion format.
    
    Args:
        strip_region_dict: Dictionary with 'left', 'top', 'right', 'bottom' keys
        orientation: 'vertical' or 'horizontal'
        angle: Rotation angle in degrees
        method: Detection method name ('canny', 'color_linear', etc.)
        confidence: Detection confidence (0.0-1.0)
        
    Returns:
        StripRegion if successful, None otherwise
    """
    if strip_region_dict is None:
        return None
    
    left = strip_region_dict.get('left', 0)
    top = strip_region_dict.get('top', 0)
    right = strip_region_dict.get('right', 0)
    bottom = strip_region_dict.get('bottom', 0)
    
    width = right - left
    height = bottom - top
    
    return StripRegion(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        width=width,
        height=height,
        confidence=confidence,
        detection_method=method,
        orientation=orientation,
        angle=angle
    )


def openai_result_to_strip_region(
    openai_result: Dict
) -> Optional[StripRegion]:
    """
    Convert OpenAI detector result to StripRegion format.
    
    Args:
        openai_result: Result from OpenAIVisionService.detect_strip()
        
    Returns:
        StripRegion if successful, None otherwise
    """
    if not openai_result.get('success'):
        return None
    
    strip_region_dict = openai_result.get('strip_region')
    if not strip_region_dict:
        return None
    
    left = strip_region_dict.get('left', 0)
    top = strip_region_dict.get('top', 0)
    right = strip_region_dict.get('right', 0)
    bottom = strip_region_dict.get('bottom', 0)
    
    width = right - left
    height = bottom - top
    
    return StripRegion(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        width=width,
        height=height,
        confidence=openai_result.get('confidence', 0.8),
        detection_method='openai',
        orientation=openai_result.get('orientation', 'vertical'),
        angle=openai_result.get('angle', 0.0)
    )


def strip_region_to_dict(strip_region: StripRegion) -> Dict:
    """
    Convert StripRegion to legacy dictionary format for backward compatibility.
    
    Args:
        strip_region: StripRegion object
        
    Returns:
        Dictionary with legacy format
    """
    return {
        'left': strip_region['left'],
        'top': strip_region['top'],
        'right': strip_region['right'],
        'bottom': strip_region['bottom']
    }

