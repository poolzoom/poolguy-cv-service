"""
Strip preparation for pad detection.

Prepares the refined strip crop for YOLO pad detection and accurate color measurement.
Applies color-safe contrast enhancement, white balance, and resizing.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext


class StripPreparator(BaseRefinementMethod):
    """Prepare strip crop for pad detection with color-safe processing."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Prepare strip crop for detection.
        
        This method processes the cropped strip image (from input_region) and
        applies color-safe enhancements:
        1. CLAHE on L channel only (color-safe contrast)
        2. Gray-world white balance
        3. Mild Gaussian blur for noise reduction
        4. Resize to target width for YOLO consistency
        
        Args:
            image: Input image (BGR format) - should be rotated crop
            input_region: Region dict with final tightened boundaries
            debug: Optional debug context
            
        Returns:
            Tuple of (region_dict, metadata)
            - region_dict: Unchanged input region
            - metadata: {'prepared_image': np.ndarray, 'target_width': int, 'scale': float}
        """
        if not input_region:
            return {
                'left': 0,
                'top': 0,
                'right': image.shape[1],
                'bottom': image.shape[0]
            }, {'prepared_image': None, 'error': 'No input region provided'}
        
        # Crop to the tightened region
        x1, y1 = input_region['left'], input_region['top']
        x2, y2 = input_region['right'], input_region['bottom']
        strip_crop = image[y1:y2, x1:x2].copy()
        
        if strip_crop.size == 0:
            return input_region, {'prepared_image': None, 'error': 'Empty crop'}
        
        # Get target width from config (default 96)
        target_width = self.config.get('target_width', 96)
        
        # Prepare the strip
        prepared_image = self._prepare_strip_for_detection(strip_crop, target_width)
        
        metadata = {
            'prepared_image': prepared_image,
            'target_width': target_width,
            'original_shape': strip_crop.shape[:2],
            'prepared_shape': prepared_image.shape[:2],
            'scale': target_width / float(strip_crop.shape[1]) if strip_crop.shape[1] > 0 else 1.0
        }
        
        if debug:
            vis = image.copy()
            # Draw the region being prepared
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan
            cv2.putText(vis, f'Prepared: {prepared_image.shape[1]}x{prepared_image.shape[0]}',
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            debug.add_step('refinement_preparation', 'Strip Preparation', vis, metadata)
        
        return input_region, metadata
    
    def _prepare_strip_for_detection(
        self,
        strip_crop: np.ndarray,
        target_width: int = 96
    ) -> np.ndarray:
        """
        Prepares a refined strip crop for YOLO pad detection AND accurate color measurement.
        
        Steps:
          1. Convert to LAB and apply CLAHE ONLY on the L channel (color-safe contrast boost)
          2. Optional mild white-balance (gray world)
          3. Remove remaining background (force crop to exact bounding region)
          4. Resize to standard width for YOLO stability
        
        Args:
            strip_crop: Cropped strip image (BGR format)
            target_width: Target width for resizing (default: 96)
            
        Returns:
            Prepared strip image (BGR format)
        """
        # ---------------------------
        # 1. LAB Color-Safe Contrast Norm (critical)
        # ---------------------------
        lab = cv2.cvtColor(strip_crop, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        
        # CLAHE on L channel ONLY (does not change color, only lightness)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge((L2, A, B))
        normalized = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        
        # ---------------------------
        # 2. Gray-World White Balance (safe for relative color)
        # ---------------------------
        # Estimate per-channel gain
        b_mean, g_mean, r_mean = np.mean(normalized.reshape(-1, 3), axis=0)
        avg = (b_mean + g_mean + r_mean) / 3.0 + 1e-5
        gain_b = avg / b_mean
        gain_g = avg / g_mean
        gain_r = avg / r_mean
        gains = np.array([gain_b, gain_g, gain_r])
        
        # Apply gains (safe because it is uniform, preserves hue)
        wb = normalized.astype(np.float32)
        wb = wb * gains
        wb = np.clip(wb, 0, 255).astype(np.uint8)
        
        # ---------------------------
        # 3. OPTIONAL: Slight Gaussian blur to remove sensor noise
        # ---------------------------
        # This is safe for color and improves detection slightly.
        color_safe = cv2.GaussianBlur(wb, (3, 3), sigmaX=0.6)
        
        # ---------------------------
        # 4. Resize to standard width for YOLO consistency
        # ---------------------------
        h, w = color_safe.shape[:2]
        scale = target_width / float(w)
        resized_h = int(h * scale)
        resized = cv2.resize(color_safe, (target_width, resized_h), interpolation=cv2.INTER_LINEAR)
        
        return resized








