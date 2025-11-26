"""
Coordinate transformation utilities for PoolGuy CV Service.
Handles transformations between relative (strip image) and absolute (original image) coordinates.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from services.interfaces import PadRegion, StripRegion
import logging

logger = logging.getLogger(__name__)


def transform_pad_coordinates_to_absolute(
    pad_regions: List[PadRegion],
    strip_region: StripRegion
) -> List[PadRegion]:
    """
    Transform pad coordinates from relative (strip image) to absolute (original image).
    
    Args:
        pad_regions: List of PadRegion objects with coordinates relative to cropped strip
        strip_region: StripRegion with absolute coordinates in original image
    
    Returns:
        List of PadRegion objects with absolute coordinates
    """
    strip_left = strip_region['left']
    strip_top = strip_region['top']
    
    absolute_pads = []
    for pad in pad_regions:
        # Transform coordinates: add strip offset
        absolute_pad = PadRegion(
            pad_index=pad['pad_index'],
            x=pad['x'] + strip_left,
            y=pad['y'] + strip_top,
            width=pad['width'],
            height=pad['height'],
            left=pad['left'] + strip_left,
            top=pad['top'] + strip_top,
            right=pad['right'] + strip_left,
            bottom=pad['bottom'] + strip_top
        )
        absolute_pads.append(absolute_pad)
    
    return absolute_pads


def crop_image_to_strip(
    image: np.ndarray,
    strip_region: StripRegion
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Crop image to strip region with bounds validation.
    
    Args:
        image: Original image array (BGR format)
        strip_region: StripRegion with coordinates
    
    Returns:
        Tuple of (cropped_image, error_message)
        - cropped_image: Cropped image array or None if error
        - error_message: Error message or None if success
    """
    h, w = image.shape[:2]
    
    # Extract coordinates
    left = strip_region['left']
    top = strip_region['top']
    right = strip_region['right']
    bottom = strip_region['bottom']
    
    # Validate bounds
    if left < 0 or top < 0 or right > w or bottom > h:
        return None, f'Strip region out of bounds: ({left}, {top}, {right}, {bottom}) vs image size ({w}, {h})'
    
    if left >= right or top >= bottom:
        return None, f'Invalid strip region: left={left} >= right={right} or top={top} >= bottom={bottom}'
    
    # Crop image
    try:
        cropped = image[top:bottom, left:right]
        
        if cropped.size == 0:
            return None, 'Cropped strip image is empty'
        
        return cropped, None
    except Exception as e:
        return None, f'Error cropping image: {str(e)}'


