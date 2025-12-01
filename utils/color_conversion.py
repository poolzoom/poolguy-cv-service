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


def opencv_lab_to_standard(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """
    Convert OpenCV 8-bit LAB values to standard LAB color space.
    
    OpenCV 8-bit LAB format:
        - L: 0-255 (scaled from 0-100)
        - a: 0-255 (128 is neutral, offset from -128 to +127)
        - b: 0-255 (128 is neutral, offset from -128 to +127)
    
    Standard LAB format:
        - L: 0-100
        - a: -128 to +127
        - b: -128 to +127
    
    Args:
        L: Lightness in OpenCV format (0-255)
        a: Green-red axis in OpenCV format (0-255)
        b: Blue-yellow axis in OpenCV format (0-255)
    
    Returns:
        Tuple of (L, a, b) in standard LAB format
    """
    L_std = L * 100.0 / 255.0
    a_std = a - 128.0
    b_std = b - 128.0
    return L_std, a_std, b_std


def extract_lab_values(image: np.ndarray, region: Tuple[int, int, int, int] = None) -> Dict[str, float]:
    """
    Extract average LAB color values from image or region.
    
    Args:
        image: Image array in LAB format (OpenCV 8-bit format)
        region: Optional (x, y, width, height) region to extract from.
                If None, extracts from entire image.
    
    Returns:
        Dictionary with L, a, b values in standard LAB format:
        - L: 0-100
        - a: -128 to +127
        - b: -128 to +127
    """
    if region is not None:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
    else:
        roi = image
    
    # Calculate mean LAB values (in OpenCV 8-bit format)
    mean_lab = np.mean(roi.reshape(-1, 3), axis=0)
    
    # Convert to standard LAB format
    L_std, a_std, b_std = opencv_lab_to_standard(mean_lab[0], mean_lab[1], mean_lab[2])
    
    return {
        'L': float(L_std),
        'a': float(a_std),
        'b': float(b_std)
    }


def extract_lab_values_with_variance(
    image: np.ndarray, 
    region: Tuple[int, int, int, int] = None
) -> Dict[str, float]:
    """
    Extract LAB color values with variance (for confidence calculation).
    
    Args:
        image: Image array in LAB format (OpenCV 8-bit format)
        region: Optional (x, y, width, height) region to extract from
    
    Returns:
        Dictionary with L, a, b values in standard LAB format and their 
        standard deviations (also converted to standard LAB scale)
    """
    if region is not None:
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
    else:
        roi = image
    
    # Reshape to list of pixels
    pixels = roi.reshape(-1, 3)
    
    # Calculate mean and std dev (in OpenCV 8-bit format)
    mean_lab = np.mean(pixels, axis=0)
    std_lab = np.std(pixels, axis=0)
    
    # Convert mean to standard LAB format
    L_std, a_std, b_std = opencv_lab_to_standard(mean_lab[0], mean_lab[1], mean_lab[2])
    
    # Convert std dev to standard LAB scale
    # L std needs to be scaled by 100/255, a and b std are already in correct units
    L_std_dev = float(std_lab[0]) * 100.0 / 255.0
    a_std_dev = float(std_lab[1])  # Already in correct scale (-128 to +127 range)
    b_std_dev = float(std_lab[2])  # Already in correct scale
    
    # Calculate overall color variance (Euclidean distance in LAB space)
    # Use the scaled std devs for consistency
    color_variance = float(np.mean([L_std_dev, a_std_dev, b_std_dev]))
    
    return {
        'L': float(L_std),
        'a': float(a_std),
        'b': float(b_std),
        'L_std': L_std_dev,
        'a_std': a_std_dev,
        'b_std': b_std_dev,
        'color_variance': color_variance
    }


def normalize_white_balance(
    image: np.ndarray, 
    white_region: Tuple[int, int, int, int] = None
) -> np.ndarray:
    """
    Normalize white balance using a reference white region.
    
    Args:
        image: Image array in LAB format (OpenCV 8-bit format)
        white_region: Optional (x, y, width, height) region containing white reference.
                     If None, attempts to find white region automatically.
    
    Returns:
        White-balanced image in LAB format (OpenCV 8-bit format)
    """
    if white_region is None:
        # Use top-right corner as default white reference (common in test strips)
        h, w = image.shape[:2]
        white_region = (int(w * 0.8), int(h * 0.1), int(w * 0.15), int(h * 0.15))
    
    x, y, rw, rh = white_region
    white_ref = image[y:y+rh, x:x+rw]
    
    # Calculate reference white LAB values in OpenCV 8-bit format
    # (L: 0-255, a: 0-255 with 128=neutral, b: 0-255 with 128=neutral)
    ref_lab = np.mean(white_ref.reshape(-1, 3), axis=0)
    
    # Target white in OpenCV 8-bit format:
    # L=255 (max brightness), a=128 (neutral), b=128 (neutral)
    target_lab = np.array([255.0, 128.0, 128.0])
    
    # Calculate adjustment factors
    adjustment = target_lab - ref_lab
    
    # Apply adjustment to entire image
    normalized = image.astype(np.float32) + adjustment
    # Clip to valid 8-bit LAB range
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    logger.debug(f'White balance normalized: ref={ref_lab}, adjustment={adjustment}')
    return normalized


def rgb_to_lab_single(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert single RGB pixel to LAB (standard format).
    
    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)
    
    Returns:
        Tuple of (L, a, b) values in standard LAB format:
        - L: 0-100
        - a: -128 to +127
        - b: -128 to +127
    """
    rgb_pixel = np.uint8([[[r, g, b]]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    opencv_lab = lab_pixel[0][0].astype(float)
    return opencv_lab_to_standard(opencv_lab[0], opencv_lab[1], opencv_lab[2])


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color string.
    
    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)
    
    Returns:
        Hex color string (e.g., "#A4C639")
    """
    return f'#{r:02X}{g:02X}{b:02X}'


def extract_color_from_region(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    center_ratio: float = 0.6
) -> Dict:
    """
    Extract average color from a region, using center portion to avoid edge effects.
    
    Args:
        image: OpenCV image array in BGR format
        x: X coordinate of region (top-left)
        y: Y coordinate of region (top-left)
        width: Width of region
        height: Height of region
        center_ratio: Ratio of center area to use (0.0-1.0, default 0.6)
    
    Returns:
        Dictionary with LAB, RGB, and hex color values:
        {
            'lab': {'L': float, 'a': float, 'b': float},
            'rgb': {'r': int, 'g': int, 'b': int},
            'hex': str
        }
    """
    img_h, img_w = image.shape[:2]
    
    # Clamp region to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    width = min(width, img_w - x)
    height = min(height, img_h - y)
    
    if width <= 0 or height <= 0:
        raise ValueError(f'Invalid region dimensions after clamping: {width}x{height}')
    
    # Calculate center region to avoid edge effects
    margin_x = int(width * (1 - center_ratio) / 2)
    margin_y = int(height * (1 - center_ratio) / 2)
    
    center_x = x + margin_x
    center_y = y + margin_y
    center_w = max(1, width - 2 * margin_x)
    center_h = max(1, height - 2 * margin_y)
    
    # Extract center region
    roi = image[center_y:center_y + center_h, center_x:center_x + center_w]
    
    # Calculate average BGR color
    mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
    b, g, r = int(round(mean_bgr[0])), int(round(mean_bgr[1])), int(round(mean_bgr[2]))
    
    # Convert BGR to LAB
    L, a, b_val = rgb_to_lab_single(r, g, b)
    
    return {
        'lab': {'L': round(L, 2), 'a': round(a, 2), 'b': round(b_val, 2)},
        'rgb': {'r': r, 'g': g, 'b': b},
        'hex': rgb_to_hex(r, g, b)
    }






