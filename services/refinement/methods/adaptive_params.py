"""
Adaptive parameter calculation for edge detection.

Calculates optimal Canny and Hough parameters based on image characteristics.
"""

import cv2
import numpy as np
from typing import Dict, Tuple


class AdaptiveParameterCalculator:
    """Calculate adaptive Canny and Hough parameters based on image analysis."""
    
    def calculate_canny_params(self, gray_image: np.ndarray) -> Tuple[int, int]:
        """
        Calculate adaptive Canny thresholds based on image statistics.
        
        Uses median-based approach:
        - low = 0.66 * median
        - high = 1.33 * median
        But also considers image size and contrast.
        
        Args:
            gray_image: Grayscale image (numpy array)
            
        Returns:
            Tuple of (canny_low, canny_high) thresholds
        """
        # Calculate image statistics
        median = np.median(gray_image)
        std = np.std(gray_image)
        h, w = gray_image.shape
        
        # Base thresholds on median (robust to outliers)
        base_low = max(10, int(0.66 * median))
        base_high = min(255, int(1.33 * median))
        
        # Adjust for contrast (high std = high contrast = can use higher thresholds)
        contrast_factor = min(2.0, std / 30.0)  # Normalize std
        low = int(base_low * (1.0 + contrast_factor * 0.3))
        high = int(base_high * (1.0 + contrast_factor * 0.3))
        
        # Adjust for image size (larger images can use higher thresholds)
        size_factor = min(1.5, (h * w) / (640 * 480))  # Normalize to 640x480
        low = int(low * (1.0 + size_factor * 0.2))
        high = int(high * (1.0 + size_factor * 0.2))
        
        # Clamp to reasonable ranges
        low = np.clip(low, 10, 100)
        high = np.clip(high, 50, 255)
        
        return low, high
    
    def calculate_hough_params(
        self, 
        image_height: int, 
        image_width: int,
        edge_density: float  # Percentage of edge pixels
    ) -> Dict[str, int]:
        """
        Calculate adaptive Hough parameters based on image size and edge quality.
        
        Args:
            image_height: Height of image
            image_width: Width of image
            edge_density: Percentage of pixels that are edges (0.0-1.0)
            
        Returns:
            Dict with 'threshold', 'min_line_length', 'max_line_gap'
        """
        # Base parameters from config
        base_min_length_ratio = 0.6
        base_threshold_ratio = 0.05
        
        # Adjust min_line_length based on edge density
        # Low density = fewer edges = need longer lines
        # High density = many edges = can use shorter lines
        if edge_density < 0.05:  # Very sparse edges
            min_length_ratio = 0.4  # Lower requirement
        elif edge_density > 0.15:  # Dense edges
            min_length_ratio = 0.7  # Higher requirement (filter noise)
        else:
            min_length_ratio = base_min_length_ratio
        
        min_line_length = max(60, int(image_height * min_length_ratio))
        
        # Adjust threshold based on edge density
        # Low density = need lower threshold to detect lines
        # High density = can use higher threshold to filter noise
        if edge_density < 0.05:
            threshold_ratio = 0.02  # Lower threshold
        elif edge_density > 0.15:
            threshold_ratio = 0.08  # Higher threshold
        else:
            threshold_ratio = base_threshold_ratio
        
        threshold = max(20, int(image_height * threshold_ratio))
        
        # Adjust max_line_gap based on image size
        # Larger images can tolerate larger gaps
        max_line_gap = max(5, int(image_height * 0.01))  # 1% of height
        
        return {
            'threshold': threshold,
            'min_line_length': min_line_length,
            'max_line_gap': max_line_gap
        }








