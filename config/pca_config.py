"""
PCA rotation detection configuration.
"""

import os
from typing import Dict

# Horizontal expansion ratio (percentage of bbox width)
PCA_HORIZONTAL_EXPANSION: float = float(os.getenv('PCA_HORIZONTAL_EXPANSION', '0.10'))  # 10%

# Vertical expansion ratio (percentage of bbox height)
PCA_VERTICAL_EXPANSION: float = float(os.getenv('PCA_VERTICAL_EXPANSION', '0.05'))  # 5%

# Minimum number of foreground points required for PCA
PCA_MIN_FOREGROUND_POINTS: int = int(os.getenv('PCA_MIN_FOREGROUND_POINTS', '100'))

# Rotation angle clamp range (degrees)
PCA_MIN_ROTATION_ANGLE: float = float(os.getenv('PCA_MIN_ROTATION_ANGLE', '-45.0'))
PCA_MAX_ROTATION_ANGLE: float = float(os.getenv('PCA_MAX_ROTATION_ANGLE', '45.0'))

# Minimum rotation angle to apply (skip very small rotations)
PCA_MIN_ROTATION_THRESHOLD: float = float(os.getenv('PCA_MIN_ROTATION_THRESHOLD', '0.01'))

# Use iterative PCA refinement (run PCA twice for better accuracy)
PCA_USE_ITERATIVE: bool = os.getenv('PCA_USE_ITERATIVE', 'true').lower() == 'true'

# Padding to add when cropping strip (for pad detection - includes pad edges)
PCA_CROP_PADDING: int = int(os.getenv('PCA_CROP_PADDING', '20'))  # 20px padding


def get_pca_config() -> Dict:
    """
    Get PCA configuration dictionary.
    
    Returns:
        Dictionary with PCA configuration parameters
    """
    return {
        'horizontal_expansion': PCA_HORIZONTAL_EXPANSION,
        'vertical_expansion': PCA_VERTICAL_EXPANSION,
        'min_foreground_points': PCA_MIN_FOREGROUND_POINTS,
        'min_rotation_angle': PCA_MIN_ROTATION_ANGLE,
        'max_rotation_angle': PCA_MAX_ROTATION_ANGLE,
        'min_rotation_threshold': PCA_MIN_ROTATION_THRESHOLD,
        'use_iterative_pca': PCA_USE_ITERATIVE,
        'crop_padding': PCA_CROP_PADDING
    }

