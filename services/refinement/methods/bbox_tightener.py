"""
Bounding box tightening for strip refinement.

Tightens bounding box by clamping inward until edges stabilize or background variance exceeds threshold.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext


class BboxTightener(BaseRefinementMethod):
    """Tighten bounding box around actual strip."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Tighten bounding box using projection-based boundaries.
        
        Uses boundaries from IntensityProjector (x_left, x_right) and computes
        vertical projection for top/bottom boundaries.
        
        Args:
            image: Input image (BGR format) - should be rotated crop
            input_region: Region dict with projection boundaries from IntensityProjector
            debug: Optional debug context
            
        Returns:
            Tuple of (refined_region, metadata)
        """
        if not input_region:
            return {
                'left': 0,
                'top': 0,
                'right': image.shape[1],
                'bottom': image.shape[0]
            }, {}
        
        # Get projection boundaries from IntensityProjector (stored in region dict)
        x_left = input_region.get('projection_x_left', 0)
        x_right = input_region.get('projection_x_right', image.shape[1] - 1)
        
        # Get top/bottom boundaries from IntensityProjector (may be None if not found)
        y_top = input_region.get('projection_y_top', None)
        y_bottom = input_region.get('projection_y_bottom', None)
        
        # Apply optional small padding (2-4 px max) to left/right only
        padding = min(4, max(2, self.config.get('padding', 3)))
        x_left = max(0, x_left - padding)
        x_right = min(image.shape[1] - 1, x_right + padding)
        
        # Use original crop area for top/bottom if not found in projection
        if y_top is None:
            y_top = input_region.get('top', 0)
        if y_bottom is None:
            y_bottom = input_region.get('bottom', image.shape[0])
        
        # Get original dimensions for metadata
        original_height = image.shape[0]
        original_width = image.shape[1]
        
        # Calculate dimensions
        tightened_height = y_bottom - y_top
        tightened_width = x_right - x_left
        
        # Never expand outside working_image bounds
        x_left = max(0, min(x_left, image.shape[1] - 1))
        x_right = max(x_left + 1, min(x_right, image.shape[1]))
        y_top = max(0, min(y_top, image.shape[0] - 1))
        y_bottom = max(y_top + 1, min(y_bottom, image.shape[0]))
        
        # Create refined region (in crop-relative coordinates)
        refined_region = {
            'left': x_left,
            'top': y_top,
            'right': x_right,
            'bottom': y_bottom
        }
        
        metadata = {
            'x_left': x_left,
            'x_right': x_right,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'tightened_width': tightened_width,
            'tightened_height': tightened_height,
            'original_width': original_width,
            'original_height': original_height,
            'width_ratio': tightened_width / tightened_height if tightened_height > 0 else 0,
            'height_retention': tightened_height / original_height if original_height > 0 else 0
        }
        
        if debug:
            vis = image.copy()
            # Draw original region if available
            orig_left = input_region.get('left', 0)
            orig_top = input_region.get('top', 0)
            orig_right = input_region.get('right', image.shape[1])
            orig_bottom = input_region.get('bottom', image.shape[0])
            cv2.rectangle(vis, (orig_left, orig_top), (orig_right, orig_bottom), (255, 255, 0), 2)  # Yellow for original
            # Draw tightened region
            cv2.rectangle(vis, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)  # Green for tightened
            cv2.putText(vis, f'Tightened: W={tightened_width}, H={tightened_height}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            debug.add_step('bbox_tightening', 'Bounding Box Tightening', vis, metadata)
        
        return refined_region, metadata
    

