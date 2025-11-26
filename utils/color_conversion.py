"""
Color space conversion utilities for PoolGuy CV Service.
Handles conversions between BGR, RGB, and LAB color spaces.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    
    Args:
        image: OpenCV image array in BGR format
    
    Returns:
        Image array in RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to LAB color space.
    
    Args:
        image: Image array in RGB format
    
    Returns:
        Image array in LAB format
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image directly to LAB color space.
    
    Args:
        image: OpenCV image array in BGR format
    
    Returns:
        Image array in LAB format
    """
    rgb_image = bgr_to_rgb(image)
    return rgb_to_lab(rgb_image)


def extract_lab_values(image: np.ndarray, region: Tuple[int, int, int, int] = None) -> Dict[str, float]:
    """
    Extract average LAB color values from image or region.
    
    Args:
        image: Image array in LAB format
        region: Optional (x, y, width, height) region to extract from.
                If None, extracts from entire image.
    
    Returns:
        Dictionary with L, a, b values
    """
    if region is not None:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
    else:
        roi = image
    
    # Calculate mean LAB values
    mean_lab = np.mean(roi.reshape(-1, 3), axis=0)
    
    return {
        'L': float(mean_lab[0]),
        'a': float(mean_lab[1]),
        'b': float(mean_lab[2])
    }


def extract_lab_values_with_variance(
    image: np.ndarray, 
    region: Tuple[int, int, int, int] = None
) -> Dict[str, float]:
    """
    Extract LAB color values with variance (for confidence calculation).
    
    Args:
        image: Image array in LAB format
        region: Optional (x, y, width, height) region to extract from
    
    Returns:
        Dictionary with L, a, b values and their standard deviations
    """
    if region is not None:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
    else:
        roi = image
    
    # Reshape to list of pixels
    pixels = roi.reshape(-1, 3)
    
    # Calculate mean and std dev
    mean_lab = np.mean(pixels, axis=0)
    std_lab = np.std(pixels, axis=0)
    
    # Calculate overall color variance (Euclidean distance in LAB space)
    color_variance = float(np.mean(std_lab))
    
    return {
        'L': float(mean_lab[0]),
        'a': float(mean_lab[1]),
        'b': float(mean_lab[2]),
        'L_std': float(std_lab[0]),
        'a_std': float(std_lab[1]),
        'b_std': float(std_lab[2]),
        'color_variance': color_variance
    }


def normalize_white_balance(
    image: np.ndarray, 
    white_region: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """
    Normalize white balance using a reference white region.
    
    Args:
        image: Image array in LAB format
        white_region: Optional (x, y, width, height) region containing white reference.
                     If None, attempts to find white region automatically.
    
    Returns:
        White-balanced image in LAB format
    """
    if white_region is None:
        # Use top-right corner as default white reference (common in test strips)
        h, w = image.shape[:2]
        white_region = (int(w * 0.8), int(h * 0.1), int(w * 0.15), int(h * 0.15))
    
    x, y, rw, rh = white_region
    white_ref = image[y:y+rh, x:x+rw]
    
    # Calculate reference white LAB values (should be high L, low a, low b)
    ref_lab = np.mean(white_ref.reshape(-1, 3), axis=0)
    
    # Target white (L=100, a=0, b=0 in LAB)
    target_lab = np.array([100.0, 0.0, 0.0])
    
    # Calculate adjustment factors
    adjustment = target_lab - ref_lab
    
    # Apply adjustment to entire image
    normalized = image.astype(np.float32) + adjustment
    normalized = np.clip(normalized, 0, 100).astype(np.uint8)
    
    logger.debug(f'White balance normalized: ref={ref_lab}, adjustment={adjustment}')
    return normalized


def rgb_to_lab_single(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert single RGB pixel to LAB.
    
    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)
    
    Returns:
        Tuple of (L, a, b) values
    """
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    return tuple(lab_pixel[0][0].astype(float))




