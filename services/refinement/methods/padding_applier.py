"""
Padding application for strip refinement.

Applies controlled padding after tightening to ensure warping sees the whole strip.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext


class PaddingApplier(BaseRefinementMethod):
    """Apply controlled padding to refined region."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Apply padding to region.
        
        Args:
            image: Input image (BGR format)
            input_region: Region dict to pad
            debug: Optional debug context
            
        Returns:
            Tuple of (padded_region, metadata)
        """
        if not input_region:
            return {
                'left': 0,
                'top': 0,
                'right': image.shape[1],
                'bottom': image.shape[0]
            }, {}
        
        h, w = image.shape[:2]
        
        # Get padding amount
        if self.config.get('adaptive', True):
            padding = self._calculate_adaptive_padding(input_region, image)
        else:
            padding = self.config.get('pixels', 10)
        
        # Apply padding with bounds checking
        padded_region = {
            'left': max(0, input_region['left'] - padding),
            'top': max(0, input_region['top'] - padding),
            'right': min(w, input_region['right'] + padding),
            'bottom': min(h, input_region['bottom'] + padding)
        }
        
        metadata = {
            'padding_applied': padding,
            'original_size': {
                'width': input_region['right'] - input_region['left'],
                'height': input_region['bottom'] - input_region['top']
            },
            'padded_size': {
                'width': padded_region['right'] - padded_region['left'],
                'height': padded_region['bottom'] - padded_region['top']
            }
        }
        
        if debug:
            vis = image.copy()
            cv2.rectangle(vis, (input_region['left'], input_region['top']),
                         (input_region['right'], input_region['bottom']), (255, 255, 0), 2)  # Original
            cv2.rectangle(vis, (padded_region['left'], padded_region['top']),
                         (padded_region['right'], padded_region['bottom']), (0, 255, 0), 2)  # Padded
            cv2.putText(vis, f'Padding: {padding}px',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            debug.add_step('padding', 'Padding Applied', vis, metadata)
        
        return padded_region, metadata
    
    def _calculate_adaptive_padding(self, region: Dict, image: np.ndarray) -> int:
        """
        Calculate adaptive padding based on region size and image resolution.
        
        Args:
            region: Region dict
            image: Full image
            
        Returns:
            Padding amount in pixels
        """
        region_width = region['right'] - region['left']
        region_height = region['bottom'] - region['top']
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Base padding on region size (larger regions need more padding)
        # But also consider image resolution
        min_padding = self.config.get('min_padding', 6)
        max_padding = self.config.get('max_padding', 12)
        
        # Scale padding based on region size (as percentage)
        region_size_ratio = (region_width * region_height) / (image_width * image_height)
        
        # Larger regions get proportionally more padding
        base_padding = min_padding + (max_padding - min_padding) * region_size_ratio
        
        # Clamp to min/max
        padding = int(np.clip(base_padding, min_padding, max_padding))
        
        return padding

