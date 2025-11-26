"""
PCA rotation detection steps for strip detection pipeline.

Provides functions for detecting and applying rotation using PCA analysis.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from services.utils.debug import DebugContext

# Optional sklearn for PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available for PCA rotation detection")

logger = logging.getLogger(__name__)


def detect_rotation_with_pca(
    image: np.ndarray,
    bbox: Dict,
    config: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Detect rotation angle using PCA on foreground pixels.
    
    Args:
        image: Full image (BGR format)
        bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2'
        config: Optional PCA configuration dict
    
    Returns:
        Tuple of (rotation_angle, pca_params):
        - rotation_angle: Rotation angle in degrees (positive = counterclockwise)
        - pca_params: Dictionary with PCA analysis details
    """
    if not SKLEARN_AVAILABLE:
        return 0.0, {'error': 'sklearn not available'}
    
    cfg = config or {}
    expand_x_ratio = cfg.get('horizontal_expansion', 0.10)
    expand_y_ratio = cfg.get('vertical_expansion', 0.05)
    min_points = cfg.get('min_foreground_points', 100)
    
    # Crop to bbox region with expansion
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    h, w = image.shape[:2]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expand_x = int(bbox_width * expand_x_ratio)
    expand_y = int(bbox_height * expand_y_ratio)
    
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, {'error': 'Empty crop', 'crop_size': (0, 0)}
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Get foreground pixels
    foreground_points = np.column_stack(np.where(binary > 0))
    threshold_method = 'otsu'
    
    if len(foreground_points) < min_points:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        foreground_points = np.column_stack(np.where(binary > 0))
        threshold_method = 'adaptive'
    
    if len(foreground_points) < min_points:
        return 0.0, {
            'foreground_points_count': len(foreground_points),
            'method': 'threshold_failed',
            'threshold_method': threshold_method
        }
    
    # Center points and apply PCA
    center = foreground_points.mean(axis=0)
    centered_points = foreground_points - center
    
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    
    pc1 = pca.components_[0]
    angle_rad = np.arctan2(pc1[0], pc1[1])
    angle_deg = np.degrees(angle_rad)
    
    # Calculate rotation based on aspect ratio
    crop_h, crop_w = crop.shape[:2]
    aspect_ratio = crop_h / crop_w if crop_w > 0 else 0
    
    if aspect_ratio > 1:
        rotation_needed = angle_deg - 90.0
    else:
        rotation_needed = -angle_deg
    
    # Handle large angles
    if abs(rotation_needed) > 45:
        if aspect_ratio > 1:
            rotation_needed = -angle_deg
        else:
            rotation_needed = angle_deg - 90.0
    
    # Clamp to config range
    min_angle = cfg.get('min_rotation_angle', -45.0)
    max_angle = cfg.get('max_rotation_angle', 45.0)
    rotation_needed = max(min_angle, min(max_angle, rotation_needed))
    
    pca_params = {
        'foreground_points_count': len(foreground_points),
        'threshold_method': threshold_method,
        'angle_deg': float(angle_deg),
        'aspect_ratio': float(aspect_ratio),
        'rotation_needed': float(rotation_needed),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist()
    }
    
    return rotation_needed, pca_params


def visualize_pca_detection(
    debug: DebugContext,
    image: np.ndarray,
    bbox: Dict,
    rotation_angle: float,
    pca_params: Dict,
    config: Dict,
    step_id: str = "01_03_pca"
):
    """
    Visualize PCA rotation detection.
    
    Args:
        debug: Debug context
        image: Original image
        bbox: YOLO bbox
        rotation_angle: Detected rotation angle
        pca_params: PCA analysis parameters
        config: PCA config
        step_id: Debug step ID
    """
    bbox_width = bbox['x2'] - bbox['x1']
    bbox_height = bbox['y2'] - bbox['y1']
    expand_x = int(bbox_width * config.get('horizontal_expansion', 0.10))
    expand_y = int(bbox_height * config.get('vertical_expansion', 0.05))
    
    expanded_x1 = max(0, bbox['x1'] - expand_x)
    expanded_y1 = max(0, bbox['y1'] - expand_y)
    expanded_x2 = min(image.shape[1], bbox['x2'] + expand_x)
    expanded_y2 = min(image.shape[0], bbox['y2'] + expand_y)
    
    # Create binary visualization
    crop_expanded = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    gray = cv2.cvtColor(crop_expanded, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    vis_pca = image.copy()
    vis_pca[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = binary_colored
    
    cv2.rectangle(vis_pca, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 2)
    cv2.rectangle(vis_pca, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 255, 0), 2)
    cv2.putText(vis_pca, f'PCA Angle: {rotation_angle:.2f}°', (expanded_x1, expanded_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    debug.add_step(
        step_id,
        'PCA Rotation Detection',
        vis_pca,
        {'rotation_angle': rotation_angle, 'pca_params': pca_params, 'bbox_used': bbox},
        f'PCA detected rotation angle of {rotation_angle:.2f}°'
    )


def visualize_rotation(
    debug: DebugContext,
    rotated_image: np.ndarray,
    rotation_angle: float,
    original_shape: Tuple[int, int],
    step_id: str = "01_04_rotated",
    title: str = "Rotated Image"
):
    """
    Visualize rotated image.
    
    Args:
        debug: Debug context
        rotated_image: Rotated image
        rotation_angle: Applied rotation angle
        original_shape: Original image shape (h, w)
        step_id: Debug step ID
        title: Step title
    """
    vis = rotated_image.copy()
    h_rot, w_rot = rotated_image.shape[:2]
    center_x = w_rot // 2
    
    cv2.line(vis, (center_x, 0), (center_x, h_rot), (0, 255, 0), 2)
    cv2.putText(vis, f'Rotated {rotation_angle:.2f}°', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    debug.add_step(
        step_id,
        title,
        vis,
        {
            'rotation_angle': rotation_angle,
            'original_size': {'width': original_shape[1], 'height': original_shape[0]},
            'rotated_size': {'width': w_rot, 'height': h_rot}
        },
        f'Image rotated by {rotation_angle:.2f}°'
    )

