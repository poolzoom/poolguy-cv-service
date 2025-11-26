"""
Visualization utilities for strip and pad detection.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def visualize_background_estimation(
    image: np.ndarray,
    sample_regions: List[Tuple[int, int, int, int]],
    bg_color: Dict
) -> np.ndarray:
    """Visualize background color estimation."""
    vis = image.copy()
    
    # Draw sample regions
    for x, y, w, h in sample_regions:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw background color swatch
    swatch_size = 50
    bg_bgr = np.array([bg_color['bgr'][2], bg_color['bgr'][1], bg_color['bgr'][0]], dtype=np.uint8)
    cv2.rectangle(vis, (10, 10), (10 + swatch_size, 10 + swatch_size), bg_bgr.tolist(), -1)
    cv2.rectangle(vis, (10, 10), (10 + swatch_size, 10 + swatch_size), (255, 255, 255), 2)
    
    cv2.putText(vis, 'Background', (10, 10 + swatch_size + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def visualize_rectangles(
    image: np.ndarray,
    rectangles: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    label_prefix: str = ''
) -> np.ndarray:
    """Visualize rectangles on image."""
    vis = image.copy()
    
    for idx, rect in enumerate(rectangles):
        x, y = rect['x'], rect['y']
        w, h = rect['width'], rect['height']
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        
        if label_prefix:
            label = f'{label_prefix}{idx}'
            cv2.putText(vis, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis


def visualize_pads(
    image: np.ndarray,
    pads: List[Dict],
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """Visualize pad locations with indices."""
    vis = image.copy()
    
    for pad in pads:
        x, y = pad['x'], pad['y']
        w, h = pad['width'], pad['height']
        idx = pad.get('pad_index', 0)
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f'Pad {idx}', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis


def visualize_strip_boundaries(
    image: np.ndarray,
    strip_region: Dict,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """Visualize strip boundaries."""
    vis = image.copy()
    
    if strip_region:
        top = strip_region['top']
        bottom = strip_region['bottom']
        left = strip_region['left']
        right = strip_region['right']
        
        cv2.rectangle(vis, (left, top), (right, bottom), color, 3)
        cv2.putText(vis, 'Strip', (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    return vis


def create_final_visualization(
    image: np.ndarray,
    strip_region: Optional[Dict],
    pads: List[Dict],
    orientation: Optional[str] = None,
    angle: Optional[float] = None
) -> np.ndarray:
    """Create final visualization with all annotations."""
    vis = image.copy()
    
    # Draw strip boundaries
    if strip_region:
        vis = visualize_strip_boundaries(vis, strip_region, (0, 255, 0))
    
    # Draw pads
    vis = visualize_pads(vis, pads, (255, 0, 0))
    
    # Add text annotations
    y_offset = 30
    if orientation:
        cv2.putText(vis, f'Orientation: {orientation}', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
    
    if angle is not None:
        cv2.putText(vis, f'Angle: {angle:.1f}°', (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30
    
    cv2.putText(vis, f'Pads: {len(pads)}', (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis




