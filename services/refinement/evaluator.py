"""
Evaluator for refinement quality metrics.
"""

import cv2
import numpy as np
from typing import Dict, Optional
from services.interfaces import StripRegion


class RefinementEvaluator:
    """Evaluate refinement quality."""
    
    def evaluate(
        self,
        original_region: Dict,
        refined_region: StripRegion,
        image: np.ndarray
    ) -> Dict:
        """
        Evaluate refinement quality.
        
        Args:
            original_region: Original YOLO region dict
            refined_region: Refined StripRegion
            image: Full image
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate size reduction
        original_area = (original_region['right'] - original_region['left']) * \
                       (original_region['bottom'] - original_region['top'])
        refined_area = refined_region['width'] * refined_region['height']
        size_reduction = 1.0 - (refined_area / original_area) if original_area > 0 else 0.0
        
        metrics['size_reduction'] = size_reduction
        metrics['original_area'] = original_area
        metrics['refined_area'] = refined_area
        
        # Calculate tightness score (how well it fits)
        # Sample edge regions to check if they're background
        tightness_score = self._calculate_tightness(refined_region, image)
        metrics['tightness_score'] = tightness_score
        
        # Calculate background removal score
        bg_removal_score = self._calculate_background_removal(original_region, refined_region, image)
        metrics['background_removal_score'] = bg_removal_score
        
        # Overall quality score (weighted average)
        metrics['overall_quality'] = (
            0.4 * tightness_score +
            0.4 * bg_removal_score +
            0.2 * min(size_reduction, 0.3) / 0.3  # Prefer moderate reduction
        )
        
        return metrics
    
    def _calculate_tightness(self, region: StripRegion, image: np.ndarray) -> float:
        """
        Calculate how tightly the region fits the strip.
        
        Lower variance at edges suggests good fit.
        """
        # Sample edge pixels
        left_edge = image[region['top']:region['bottom'], region['left']:region['left']+5, :]
        right_edge = image[region['top']:region['bottom'], region['right']-5:region['right'], :]
        
        # Calculate variance (lower = tighter fit)
        left_var = np.var(left_edge)
        right_var = np.var(right_edge)
        avg_var = (left_var + right_var) / 2.0
        
        # Normalize to 0-1 score (lower variance = higher score)
        # Assuming variance range 0-10000, normalize
        tightness = max(0.0, 1.0 - (avg_var / 1000.0))
        
        return min(1.0, tightness)
    
    def _calculate_background_removal(self, original: Dict, refined: StripRegion, image: np.ndarray) -> float:
        """
        Calculate how much background was removed.
        
        Compare variance in removed regions vs strip region.
        """
        # Original region
        orig_crop = image[original['top']:original['bottom'], original['left']:original['right'], :]
        orig_var = np.var(orig_crop)
        
        # Refined region
        refined_crop = image[refined['top']:refined['bottom'], refined['left']:refined['right'], :]
        refined_var = np.var(refined_crop)
        
        # Lower variance in refined = better background removal
        if orig_var == 0:
            return 1.0
        
        # Score based on variance reduction
        reduction = (orig_var - refined_var) / orig_var if orig_var > 0 else 0.0
        
        return max(0.0, min(1.0, reduction))








