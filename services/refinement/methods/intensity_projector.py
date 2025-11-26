"""
Intensity projection for strip refinement.

Projects intensities horizontally to find dense/consistent regions.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext


class IntensityProjector(BaseRefinementMethod):
    """Project intensities horizontally and vertically to identify strip region."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None,
        yolo_constraint: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Project intensities horizontally to find left/right strip boundaries.
        
        Uses intensity projection histograms to find left/right boundaries only.
        Top/bottom boundaries use the original crop area (not detected).
        
        Does NOT crop yet - only stores boundaries in metadata for BboxTightener.
        
        Args:
            image: Input image (BGR format) - should be rotated crop (rotation already done in yolo_pca)
            input_region: Optional region dict (if None, uses full image)
            debug: Optional debug context
            yolo_constraint: Ignored - not used in deterministic algorithm
            
        Returns:
            Tuple of (region_dict, metadata)
            - region_dict: Unchanged input region with projection boundaries stored
            - metadata: {'x_left': int, 'x_right': int, 'y_top': None, 'y_bottom': None, ...}
        """
        # Crop to region if provided
        if input_region:
            x1, y1 = input_region['left'], input_region['top']
            x2, y2 = input_region['right'], input_region['bottom']
            cropped = image[y1:y2, x1:x2].copy()
        else:
            cropped = image.copy()
        
        # Convert rotated crop to grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Compute horizontal projection: mean across height for each column
        proj_x = np.mean(gray, axis=0)
        
        # Normalize projection
        proj_x_max = proj_x.max()
        if proj_x_max > 0:
            p_x = proj_x / proj_x_max
        else:
            p_x = proj_x
        
        # Compute derivative: absolute difference
        dp_x = np.abs(np.diff(p_x))
        
        # Find left boundary as argmax in first half of dp_x
        mid = len(dp_x) // 2
        first_half = dp_x[:mid]
        if len(first_half) > 0:
            x_left = int(np.argmax(first_half))
        else:
            x_left = 0
        
        # Find right boundary as argmax in second half of dp_x
        second_half = dp_x[mid:]
        if len(second_half) > 0:
            x_right = int(np.argmax(second_half) + mid)
        else:
            x_right = cropped.shape[1] - 1
        
        # Store boundaries in metadata AND in region dict for BboxTightener to access
        # Return unchanged input region but add projection boundaries
        refined_region = input_region.copy() if input_region else {
            'left': 0,
            'top': 0,
            'right': cropped.shape[1],
            'bottom': cropped.shape[0]
        }
        
        # Now compute vertical projection for top/bottom boundaries
        # Use the cropped horizontal region (between x_left and x_right) for better accuracy
        strip_crop = gray[:, x_left:x_right] if x_right > x_left else gray
        
        # Vertical projection: mean across width for each row
        proj_y = np.mean(strip_crop, axis=1)
        
        # Normalize projection
        proj_y_max = proj_y.max()
        if proj_y_max > 0:
            p_y = proj_y / proj_y_max
        else:
            p_y = proj_y
        
        # Compute derivative: absolute difference
        dp_y = np.abs(np.diff(p_y))
        
        # Find top boundary as argmax in first half of dp_y
        mid_y = len(dp_y) // 2
        first_half_y = dp_y[:mid_y]
        if len(first_half_y) > 0:
            y_top = int(np.argmax(first_half_y))
        else:
            y_top = None  # Indicate not found
        
        # Find bottom boundary as argmax in second half of dp_y
        second_half_y = dp_y[mid_y:]
        if len(second_half_y) > 0:
            y_bottom = int(np.argmax(second_half_y) + mid_y)
        else:
            y_bottom = None  # Indicate not found
        
        # Validate top/bottom boundaries - must be within 5% of edges
        crop_height = cropped.shape[0]
        top_threshold = int(crop_height * 0.05)  # 5% from top
        bottom_threshold = int(crop_height * 0.95)  # 5% from bottom (95% of height)
        
        if y_top is not None and y_top > top_threshold:
            # Top boundary too far from top edge, use original crop
            y_top = None
        
        if y_bottom is not None and y_bottom < bottom_threshold:
            # Bottom boundary too far from bottom edge, use original crop
            y_bottom = None
        
        # Store projection boundaries in region dict for BboxTightener
        refined_region['projection_x_left'] = x_left
        refined_region['projection_x_right'] = x_right
        refined_region['projection_y_top'] = y_top
        refined_region['projection_y_bottom'] = y_bottom
        
        metadata = {
            'x_left': x_left,
            'x_right': x_right,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'proj_x': proj_x,
            'p_x': p_x,
            'dp_x': dp_x,
            'proj_y': proj_y,
            'p_y': p_y,
            'dp_y': dp_y,
            'cropped_shape': cropped.shape[:2]
        }
        
        if debug:
            vis = image.copy()
            if input_region:
                x1, y1, x2, y2 = input_region['left'], input_region['top'], input_region['right'], input_region['bottom']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for original
                # Draw detected boundaries on cropped region
                cv2.line(vis, (x1 + x_left, y1), (x1 + x_left, y2), (0, 255, 0), 2)  # Green line for left
                cv2.line(vis, (x1 + x_right, y1), (x1 + x_right, y2), (0, 255, 0), 2)  # Green line for right
                if y_top is not None:
                    cv2.line(vis, (x1, y1 + y_top), (x2, y1 + y_top), (0, 255, 0), 2)  # Green line for top
                if y_bottom is not None:
                    cv2.line(vis, (x1, y1 + y_bottom), (x2, y1 + y_bottom), (0, 255, 0), 2)  # Green line for bottom
            top_str = f'T={y_top}' if y_top is not None else 'T=None'
            bottom_str = f'B={y_bottom}' if y_bottom is not None else 'B=None'
            cv2.putText(vis, f'Projection: L={x_left}, R={x_right}, {top_str}, {bottom_str}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            debug.add_step('intensity_projection', 'Intensity Projection', vis, metadata)
        
        return refined_region, metadata
    

