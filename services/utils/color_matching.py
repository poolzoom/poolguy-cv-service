"""
Color matching service for PoolGuy CV Service.
Matches extracted pad colors to reference swatches using CIEDE2000 algorithm.
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_diff import delta_e_cie2000

logger = logging.getLogger(__name__)


class ColorMatchingService:
    """Service for matching colors to reference swatches using CIEDE2000."""
    
    def __init__(self):
        """Initialize color matching service."""
        self.logger = logging.getLogger(__name__)
    
    def match_colors(
        self,
        extracted_colors: List[Dict],
        reference_swatches: List[Dict]
    ) -> Dict:
        """
        Match extracted pad colors to reference swatches.
        
        Args:
            extracted_colors: List of extracted color dicts with LAB values:
                [{'pad_index': 0, 'lab': {'L': 50.0, 'a': 0.0, 'b': 0.0}, ...}, ...]
            reference_swatches: List of reference swatch dicts:
                [{'swatch_id': 'chlorine_0', 'lab': {'L': 50.0, 'a': 0.0, 'b': 0.0}, 'chemistry_value': 0.5}, ...]
        
        Returns:
            Dictionary with matching results:
            {
                'success': bool,
                'matches': [
                    {
                        'pad_index': int,
                        'matched_swatch_id': str,
                        'chemistry_value': float,
                        'delta_e': float,
                        'color_matching_confidence': float,
                        'pad_detection_confidence': float,
                        'overall_confidence': float
                    },
                    ...
                ],
                'overall_confidence': float
            }
        """
        try:
            if not extracted_colors:
                return {
                    'success': False,
                    'error': 'No extracted colors provided',
                    'error_code': 'INVALID_PARAMETER'
                }
            
            if not reference_swatches:
                return {
                    'success': False,
                    'error': 'No reference swatches provided',
                    'error_code': 'INVALID_PARAMETER'
                }
            
            matches = []
            
            for extracted in extracted_colors:
                pad_index = extracted.get('pad_index', 0)
                extracted_lab = extracted.get('lab', {})
                pad_detection_confidence = extracted.get('pad_detection_confidence', 0.5)
                
                # Find best match
                best_match = self._find_best_match(extracted_lab, reference_swatches)
                
                if best_match:
                    # Calculate color matching confidence
                    color_matching_confidence = self._calculate_color_matching_confidence(
                        best_match['delta_e'],
                        best_match.get('second_delta_e', None)
                    )
                    
                    # Calculate overall confidence (combines both)
                    overall_confidence = (
                        pad_detection_confidence * 0.5 + 
                        color_matching_confidence * 0.5
                    )
                    
                    matches.append({
                        'pad_index': pad_index,
                        'matched_swatch_id': best_match['swatch_id'],
                        'chemistry_value': best_match['chemistry_value'],
                        'delta_e': best_match['delta_e'],
                        'color_matching_confidence': float(color_matching_confidence),
                        'pad_detection_confidence': float(pad_detection_confidence),
                        'overall_confidence': float(overall_confidence)
                    })
                else:
                    # No match found
                    matches.append({
                        'pad_index': pad_index,
                        'matched_swatch_id': None,
                        'chemistry_value': None,
                        'delta_e': None,
                        'color_matching_confidence': 0.0,
                        'pad_detection_confidence': float(pad_detection_confidence),
                        'overall_confidence': float(pad_detection_confidence * 0.5)
                    })
            
            # Calculate overall confidence across all matches
            overall_confidence = (
                sum(m['overall_confidence'] for m in matches) / len(matches)
                if matches else 0.0
            )
            
            return {
                'success': True,
                'matches': matches,
                'overall_confidence': float(overall_confidence)
            }
            
        except Exception as e:
            self.logger.error(f'Error matching colors: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'COLOR_MATCHING_FAILED'
            }
    
    def _find_best_match(
        self,
        extracted_lab: Dict,
        reference_swatches: List[Dict]
    ) -> Optional[Dict]:
        """
        Find best matching swatch for extracted color.
        
        Args:
            extracted_lab: LAB color dict {'L': float, 'a': float, 'b': float}
            reference_swatches: List of reference swatch dicts
        
        Returns:
            Dict with best match info or None
        """
        if not extracted_lab or 'L' not in extracted_lab:
            return None
        
        # Create LabColor object for extracted color
        try:
            extracted_color = LabColor(
                lab_l=extracted_lab['L'],
                lab_a=extracted_lab['a'],
                lab_b=extracted_lab['b']
            )
        except Exception as e:
            self.logger.error(f'Invalid LAB color: {extracted_lab}, error: {e}')
            return None
        
        # Calculate delta E for all swatches
        delta_e_values = []
        
        for swatch in reference_swatches:
            swatch_lab = swatch.get('lab', {})
            if not swatch_lab or 'L' not in swatch_lab:
                continue
            
            try:
                swatch_color = LabColor(
                    lab_l=swatch_lab['L'],
                    lab_a=swatch_lab['a'],
                    lab_b=swatch_lab['b']
                )
                
                # Calculate CIEDE2000 delta E
                delta_e = delta_e_cie2000(extracted_color, swatch_color)
                
                delta_e_values.append({
                    'swatch_id': swatch.get('swatch_id', ''),
                    'chemistry_value': swatch.get('chemistry_value', 0.0),
                    'delta_e': delta_e,
                    'swatch': swatch
                })
            except Exception as e:
                self.logger.warning(f'Error calculating delta E for swatch: {e}')
                continue
        
        if not delta_e_values:
            return None
        
        # Sort by delta E (lower is better)
        delta_e_values.sort(key=lambda x: x['delta_e'])
        
        # Get best match
        best_match = delta_e_values[0]
        
        # Get second best for ambiguity check
        second_delta_e = delta_e_values[1]['delta_e'] if len(delta_e_values) > 1 else None
        
        return {
            'swatch_id': best_match['swatch_id'],
            'chemistry_value': best_match['chemistry_value'],
            'delta_e': best_match['delta_e'],
            'second_delta_e': second_delta_e
        }
    
    def _calculate_color_matching_confidence(
        self,
        delta_e: float,
        second_delta_e: Optional[float] = None
    ) -> float:
        """
        Calculate color matching confidence based on delta E distance.
        
        Primary factor (70%): ΔE distance to nearest reference swatch
        Secondary factor (30%): Distance to second-closest swatch (ambiguity check)
        
        Args:
            delta_e: Delta E distance to best match
            second_delta_e: Delta E distance to second-best match (optional)
        
        Returns:
            Confidence score (0-1)
        """
        primary_weight = 0.7
        secondary_weight = 0.3
        
        # Primary factor: delta E distance
        # CIEDE2000 interpretation:
        # < 1: Not perceptible
        # 1-2: Perceptible through close observation
        # 2-10: Perceptible at a glance
        # 10-49: Colors are more similar than opposite
        # > 49: Colors are opposite
        
        # Convert delta E to confidence (lower delta E = higher confidence)
        # Use exponential decay: confidence = e^(-delta_e / threshold)
        # Threshold of 10 gives good distribution
        threshold = 10.0
        primary_score = max(0.0, min(1.0, np.exp(-delta_e / threshold)))
        
        # Secondary factor: ambiguity check
        # If second match is very close, confidence is lower
        if second_delta_e is not None:
            ambiguity = abs(delta_e - second_delta_e)
            # If ambiguity is small (< 2), confidence is reduced
            if ambiguity < 2.0:
                ambiguity_penalty = ambiguity / 2.0  # 0-1 penalty
                secondary_score = 1.0 - (ambiguity_penalty * 0.5)  # Max 50% reduction
            else:
                secondary_score = 1.0  # Clear winner
        else:
            secondary_score = 1.0  # No ambiguity if only one match
        
        # Weighted combination
        confidence = (primary_weight * primary_score) + (secondary_weight * secondary_score)
        
        # Ensure 0-1 range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence

