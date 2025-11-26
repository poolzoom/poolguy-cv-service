"""
Edge detection for strip refinement.

Identifies strip edges using gradient magnitude, white backing, or color variance.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext


class EdgeDetector(BaseRefinementMethod):
    """Detect strip edges using multiple methods."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Passthrough for projection-based edges.
        
        This method no longer performs edge detection. It forwards the input region
        and metadata from IntensityProjector to BboxTightener.
        
        Args:
            image: Input image (BGR format)
            input_region: Region dict from IntensityProjector
            debug: Optional debug context
            
        Returns:
            Tuple of (region_dict, metadata)
            - region_dict: Unchanged input region (passthrough)
            - metadata: Empty dict (edges will be applied in BboxTightener)
        """
        # Passthrough - no edge detection, just forward the region
        refined_region = input_region.copy() if input_region else {
            'left': 0,
            'top': 0,
            'right': image.shape[1],
            'bottom': image.shape[0]
        }
        
        metadata = {
            'passthrough': True,
            'note': 'Edge detection bypassed - using projection-based boundaries from IntensityProjector'
        }
        
        if debug:
            vis = image.copy()
            if input_region:
                x1, y1, x2, y2 = input_region['left'], input_region['top'], input_region['right'], input_region['bottom']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for input
            cv2.putText(vis, 'Edge Detection (Passthrough)',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            debug.add_step('edge_detection', 'Edge Detection (Passthrough)', vis, metadata)
        
        return refined_region, metadata

