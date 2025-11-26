"""
Configuration for strip refinement parameters.

Allows easy adjustment of refinement parameters for iterative development.
"""

import os
import numpy as np
from typing import Dict, Optional

# Base configuration
REFINEMENT_CONFIG: Dict = {
    'orientation': {
        'method': os.getenv('REFINEMENT_ORIENTATION_METHOD', 'hough'),  # 'hough' or 'minarearect'
        'max_rotation_angle': float(os.getenv('REFINEMENT_MAX_ROTATION_ANGLE', '30.0')),  # Max rotation in degrees
        'rotation_combination': os.getenv('REFINEMENT_ROTATION_COMBINATION', 'extreme'),  # 'average', 'extreme', or 'weighted'
        
        # Canny edge detection parameters
        'canny_low': int(os.getenv('REFINEMENT_CANNY_LOW', '30')),  # Lower = more edges detected
        'canny_high': int(os.getenv('REFINEMENT_CANNY_HIGH', '100')),  # Lower ratio = better continuity
        'canny_blur_size': int(os.getenv('REFINEMENT_CANNY_BLUR_SIZE', '7')),  # Gaussian blur kernel size
        'canny_blur_sigma': float(os.getenv('REFINEMENT_CANNY_BLUR_SIGMA', '1.5')),  # Gaussian blur sigma
        
        # Morphological operations
        'morph_close_kernel': int(os.getenv('REFINEMENT_MORPH_CLOSE', '3')),  # Close gaps in edges
        'morph_open_kernel': int(os.getenv('REFINEMENT_MORPH_OPEN', '2')),  # Remove noise
        
        # Hough line detection parameters
        'hough_min_line_length_ratio': float(os.getenv('REFINEMENT_HOUGH_MIN_LENGTH', '0.6')),  # Min line length as ratio of height (0.6 = 60%)
        'hough_min_line_length_absolute': int(os.getenv('REFINEMENT_HOUGH_MIN_LENGTH_ABS', '60')),  # Absolute minimum in pixels
        'hough_threshold_ratio': float(os.getenv('REFINEMENT_HOUGH_THRESHOLD', '0.05')),  # Threshold as ratio of height (lower = more lines, 0.05 = 5%)
        'hough_threshold_absolute': int(os.getenv('REFINEMENT_HOUGH_THRESHOLD_ABS', '20')),  # Absolute minimum threshold
        'hough_max_line_gap': int(os.getenv('REFINEMENT_HOUGH_MAX_GAP', '15')),  # Max gap in line (pixels)
        
        # Line filtering parameters
        'angle_tolerance_degrees': float(os.getenv('REFINEMENT_ANGLE_TOLERANCE', '15.0')),  # Max deviation from vertical (degrees)
        'spatial_tolerance_ratio': float(os.getenv('REFINEMENT_SPATIAL_TOLERANCE', '0.4')),  # Edge position tolerance as ratio of width
        'use_spatial_filter': os.getenv('REFINEMENT_USE_SPATIAL_FILTER', 'true').lower() == 'true',  # Filter by expected edge position
        
        # Line selection parameters
        'lines_per_side': int(os.getenv('REFINEMENT_LINES_PER_SIDE', '3')),  # Number of best lines to consider per side
        'line_score_length_weight': float(os.getenv('REFINEMENT_LINE_LENGTH_WEIGHT', '0.6')),  # Weight for line length in scoring
        'line_score_proximity_weight': float(os.getenv('REFINEMENT_LINE_PROXIMITY_WEIGHT', '0.4')),  # Weight for proximity in scoring
        
        # Region expansion
        'expand_ratio': float(os.getenv('REFINEMENT_EXPAND_RATIO', '0.05')),  # Expand region by this ratio (5%)
        
        # Fallback parameters
        'min_contour_area': int(os.getenv('REFINEMENT_MIN_CONTOUR_AREA', '100')),  # Minimum contour area for minAreaRect fallback
        'min_rotation_threshold': float(os.getenv('REFINEMENT_MIN_ROTATION', '0.1')),  # Minimum rotation to apply (degrees)
        
        # Adaptive parameter flags
        'use_adaptive_canny': os.getenv('REFINEMENT_ADAPTIVE_CANNY', 'true').lower() == 'true',  # Use adaptive Canny thresholds
        'use_adaptive_hough': os.getenv('REFINEMENT_ADAPTIVE_HOUGH', 'true').lower() == 'true',  # Use adaptive Hough parameters
        'use_iterative_retry': os.getenv('REFINEMENT_ITERATIVE_RETRY', 'true').lower() == 'true',  # Retry with relaxed params if detection fails
    },
    'projection': {
        'window_size': int(os.getenv('REFINEMENT_PROJECTION_WINDOW', '5')),
        'smoothing': os.getenv('REFINEMENT_PROJECTION_SMOOTHING', 'gaussian'),  # 'gaussian' or 'none'
        'sigma': float(os.getenv('REFINEMENT_PROJECTION_SIGMA', '2.0')),
        'yolo_search_margin_ratio': float(os.getenv('REFINEMENT_YOLO_SEARCH_MARGIN', '0.1'))  # 10% margin around YOLO bbox
    },
    'edge_detection': {
        'gradient_threshold': float(os.getenv('REFINEMENT_GRADIENT_THRESHOLD', '0.3')),
        'white_backing_threshold': int(os.getenv('REFINEMENT_WHITE_THRESHOLD', '200')),
        'color_variance_threshold': float(os.getenv('REFINEMENT_COLOR_VARIANCE', '15.0'))
    },
    'tightening': {
        'edge_stability_threshold': float(os.getenv('REFINEMENT_EDGE_STABILITY', '0.05')),
        'background_variance_threshold': float(os.getenv('REFINEMENT_BG_VARIANCE', '20.0')),
        'white_background_threshold': int(os.getenv('REFINEMENT_WHITE_BG', '240'))
    },
    'padding': {
        'pixels': int(os.getenv('REFINEMENT_PADDING', '10')),
        'adaptive': os.getenv('REFINEMENT_ADAPTIVE_PADDING', 'true').lower() == 'true',
        'min_padding': int(os.getenv('REFINEMENT_MIN_PADDING', '6')),
        'max_padding': int(os.getenv('REFINEMENT_MAX_PADDING', '12'))
    }
}

# Preset configurations
PRESETS: Dict[str, Dict] = {
    'conservative': {
        'tightening': {
            'edge_stability_threshold': 0.1,
            'background_variance_threshold': 30.0,
            'white_background_threshold': 230
        },
        'padding': {
            'pixels': 12,
            'adaptive': True,
            'min_padding': 8,
            'max_padding': 15
        }
    },
    'aggressive': {
        'tightening': {
            'edge_stability_threshold': 0.02,
            'background_variance_threshold': 10.0,
            'white_background_threshold': 250
        },
        'padding': {
            'pixels': 6,
            'adaptive': True,
            'min_padding': 4,
            'max_padding': 8
        }
    },
    'balanced': REFINEMENT_CONFIG.copy()
}


def get_config(preset: Optional[str] = None) -> Dict:
    """
    Get refinement configuration.
    
    Args:
        preset: Preset name ('conservative', 'aggressive', 'balanced') or None for default
        
    Returns:
        Configuration dictionary
    """
    if preset and preset in PRESETS:
        # Merge preset with base config
        config = REFINEMENT_CONFIG.copy()
        for key, value in PRESETS[preset].items():
            if isinstance(value, dict):
                config[key] = {**config.get(key, {}), **value}
            else:
                config[key] = value
        return config
    return REFINEMENT_CONFIG.copy()

