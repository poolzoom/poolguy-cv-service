"""
Image loading utilities for PoolGuy CV Service.
Supports loading images from local paths and S3 signed URLs.
"""

import cv2
import numpy as np
import requests
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def load_image(image_path: str, timeout: int = 30) -> np.ndarray:
    """
    Load image from local path or URL (S3 signed URL).
    
    Args:
        image_path: Path to image (local file path or HTTP/HTTPS URL)
        timeout: Request timeout in seconds for URL downloads
    
    Returns:
        OpenCV image array in BGR format (numpy.ndarray)
    
    Raises:
        ValueError: If image path is invalid or image cannot be loaded
        requests.RequestException: If URL request fails
    """
    if not image_path:
        raise ValueError('image_path cannot be empty')
    
    # Check if it's a URL
    parsed = urlparse(image_path)
    is_url = parsed.scheme in ('http', 'https')
    
    try:
        if is_url:
            logger.info(f'Loading image from URL: {image_path}')
            response = requests.get(image_path, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Read image from response content
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            logger.info(f'Loading image from local path: {image_path}')
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f'Failed to decode image: {image_path}')
        
        # Validate image dimensions
        if image.size == 0:
            raise ValueError(f'Image is empty: {image_path}')
        
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            raise ValueError(f'Image too small: {width}x{height} pixels')
        
        logger.debug(f'Successfully loaded image: {width}x{height} pixels')
        return image
        
    except requests.RequestException as e:
        logger.error(f'Failed to download image from URL: {e}')
        raise ValueError(f'Failed to load image from URL: {str(e)}')
    except Exception as e:
        logger.error(f'Unexpected error loading image: {e}', exc_info=True)
        raise ValueError(f'Failed to load image: {str(e)}')


def validate_image_format(image_path: str) -> bool:
    """
    Validate that image path has a supported format.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    return any(image_path.lower().endswith(ext) for ext in supported_formats)


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: OpenCV image array
    
    Returns:
        Dictionary with image information (width, height, channels, dtype)
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes
    }




