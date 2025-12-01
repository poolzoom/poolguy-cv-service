"""
Color mapping service for bottle pipeline.
Maps detected pad colors to reference squares and calculates mapped values.

DEPRECATED: This service is no longer used in the new stateless API.
Color mapping is now handled by the Laravel application after extracting colors.
Kept for backwards compatibility only.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

from services.interfaces import LabColor, ReferenceSquare, BottlePadRegion
from services.utils.color_matching import ColorMatchingService
from utils.color_conversion import bgr_to_lab, extract_lab_values

logger = logging.getLogger(__name__)


class ColorMappingService:
    """
    Service for mapping detected pad colors to reference squares.
    
    Uses CIEDE2000 color distance to match pad colors to reference ranges.
    """
    
    def __init__(self):
        """Initialize color mapping service."""
        self.logger = logging.getLogger(__name__)
        self.color_matcher = ColorMatchingService()
    
    def map_colors(
        self,
        image: np.ndarray,
        pads: List[Dict],
        reference_squares: List[ReferenceSquare],
        pad_names: Optional[List[Dict]] = None,
        debug: Optional[object] = None
    ) -> Dict:
        """
        Map detected pad colors to reference squares and calculate values.
        
        Args:
            image: Input image (BGR format)
            pads: List of detected pad regions with coordinates
            reference_squares: List of reference color squares
            pad_names: Optional list of pad names from text extraction
            debug: Optional debug context
        
        Returns:
            Dictionary with mapped results:
            {
                'success': bool,
                'mapped_pads': [
                    {
                        'pad_index': int,
                        'name': Optional[str],
                        'region': PadRegion,
                        'reference_range': Optional[str],
                        'reference_squares': List[ReferenceSquare],
                        'detected_color': LabColor,
                        'mapped_value': Optional[str],
                        'confidence': float
                    },
                    ...
                ],
                'overall_confidence': float
            }
        """
        try:
            # Convert image to LAB for color extraction
            lab_image = bgr_to_lab(image)
            
            # Extract colors from each pad (or use from text extraction if available)
            pad_colors = []
            for pad in pads:
                pad_index = pad.get('pad_index', 0)
                
                # Check if we have LAB color from text extraction for this pad
                lab_color_from_text = None
                if pad_names:
                    for name_data in pad_names:
                        # Match by pad_region if available
                        pad_region = name_data.get('pad_region', {})
                        if pad_region:
                            pad_x = pad.get('x', pad.get('left', 0))
                            pad_y = pad.get('y', pad.get('top', 0))
                            name_pad_x = pad_region.get('x', 0)
                            name_pad_y = pad_region.get('y', 0)
                            
                            # Check if regions overlap or are close
                            distance = ((pad_x - name_pad_x) ** 2 + (pad_y - name_pad_y) ** 2) ** 0.5
                            if distance < 100:  # Within 100 pixels
                                lab_color_from_text = name_data.get('lab_color')
                                break
                
                # Use LAB color from text extraction if available, otherwise extract from image
                if lab_color_from_text:
                    lab_color = lab_color_from_text
                else:
                    x = pad.get('x', pad.get('left', 0))
                    y = pad.get('y', pad.get('top', 0))
                    w = pad.get('width', 0)
                    h = pad.get('height', 0)
                    
                    if w > 0 and h > 0:
                        lab_color = extract_lab_values(lab_image, (x, y, w, h))
                    else:
                        continue
                
                pad_colors.append({
                    'pad_index': pad_index,
                    'lab': lab_color
                })
            
            # Match pad colors to reference squares
            mapped_pads = []
            
            for pad in pads:
                pad_index = pad.get('pad_index', 0)
                
                # Find pad color
                pad_color_data = next(
                    (pc for pc in pad_colors if pc['pad_index'] == pad_index),
                    None
                )
                
                if not pad_color_data:
                    continue
                
                detected_color: LabColor = pad_color_data['lab']
                
                # Find pad name from text extraction
                pad_name = None
                reference_range = None
                pad_lab_color_from_text = None
                closest_name = None
                
                if pad_names:
                    # Try to match pad by pad_region position (more accurate than text region)
                    pad_x = pad.get('x', pad.get('left', 0))
                    pad_y = pad.get('y', pad.get('top', 0))
                    pad_w = pad.get('width', 0)
                    pad_h = pad.get('height', 0)
                    
                    min_distance = float('inf')
                    
                    for name_data in pad_names:
                        # Prefer matching by pad_region if available
                        pad_region = name_data.get('pad_region', {})
                        if pad_region and pad_region.get('width', 0) > 0:
                            # Calculate IoU or center distance for pad regions
                            name_pad_x = pad_region.get('x', 0)
                            name_pad_y = pad_region.get('y', 0)
                            name_pad_w = pad_region.get('width', 0)
                            name_pad_h = pad_region.get('height', 0)
                            
                            # Calculate center points
                            pad_center_x = pad_x + pad_w / 2
                            pad_center_y = pad_y + pad_h / 2
                            name_center_x = name_pad_x + name_pad_w / 2
                            name_center_y = name_pad_y + name_pad_h / 2
                            
                            distance = ((pad_center_x - name_center_x) ** 2 + (pad_center_y - name_center_y) ** 2) ** 0.5
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_name = name_data
                        else:
                            # Fallback to text region
                            name_region = name_data.get('region', {})
                            name_x = name_region.get('x', 0)
                            name_y = name_region.get('y', 0)
                            
                            distance = ((pad_x - name_x) ** 2 + (pad_y - name_y) ** 2) ** 0.5
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_name = name_data
                    
                    # Use a more lenient threshold since pad_region should be more accurate
                    threshold = 300 if closest_name and closest_name.get('pad_region') else 200
                    if closest_name and min_distance < threshold:
                        pad_name = closest_name.get('name')
                        reference_range = closest_name.get('reference_range')
                        pad_lab_color_from_text = closest_name.get('lab_color')
                        
                        # If we have LAB color from text extraction, prefer it over detected
                        if pad_lab_color_from_text:
                            detected_color = pad_lab_color_from_text
                
                # Use reference values from text extraction if available
                pad_reference_squares = []
                if closest_name and 'reference_values' in closest_name:
                    # Convert reference_values from text extraction to ReferenceSquare format
                    ref_values = closest_name.get('reference_values', [])
                    for ref_val in ref_values:
                        if not isinstance(ref_val, dict):
                            continue
                        
                        value = ref_val.get('value', '')
                        ref_region = ref_val.get('region', {})
                        ref_lab = ref_val.get('lab_color')
                        
                        if value and ref_lab and isinstance(ref_lab, dict):
                            from services.interfaces import ReferenceSquare
                            ref_square: ReferenceSquare = {
                                'color': {
                                    'L': float(ref_lab.get('L', 0)),
                                    'a': float(ref_lab.get('a', 0)),
                                    'b': float(ref_lab.get('b', 0))
                                },
                                'value': value,
                                'region': {
                                    'pad_index': pad_index,
                                    'x': int(ref_region.get('x', 0)),
                                    'y': int(ref_region.get('y', 0)),
                                    'width': int(ref_region.get('width', 0)),
                                    'height': int(ref_region.get('height', 0)),
                                    'left': int(ref_region.get('x', 0)),
                                    'top': int(ref_region.get('y', 0)),
                                    'right': int(ref_region.get('x', 0) + ref_region.get('width', 0)),
                                    'bottom': int(ref_region.get('y', 0) + ref_region.get('height', 0))
                                },
                                'confidence': 0.9,  # High confidence from text extraction
                                'associated_pad': pad_index
                            }
                            pad_reference_squares.append(ref_square)
                
                # Fallback to detected reference squares if text extraction didn't provide them
                if not pad_reference_squares:
                    pad_reference_squares = [
                        ref for ref in reference_squares
                        if ref.get('associated_pad') == pad_index or
                        (not ref.get('associated_pad') and len(reference_squares) <= len(pads))
                    ]
                    
                    # If no specific association, try to match by proximity
                    if not pad_reference_squares and reference_squares:
                        # Group reference squares by proximity to pads
                        pad_x = pad.get('x', pad.get('left', 0))
                        pad_y = pad.get('y', pad.get('top', 0))
                        
                        for ref_square in reference_squares:
                            ref_region = ref_square.get('region', {})
                            ref_x = ref_region.get('x', 0)
                            ref_y = ref_region.get('y', 0)
                            
                            distance = ((pad_x - ref_x) ** 2 + (pad_y - ref_y) ** 2) ** 0.5
                            
                            if distance < 300:  # Within 300 pixels
                                pad_reference_squares.append(ref_square)
                
                # Map color to reference squares
                mapped_value = None
                mapping_confidence = 0.0
                
                if pad_reference_squares:
                    # Use color matching service to find best match
                    reference_swatches = [
                        {
                            'swatch_id': f"ref_{i}",
                            'lab': ref_square['color'],
                            'chemistry_value': ref_square.get('value', '')
                        }
                        for i, ref_square in enumerate(pad_reference_squares)
                    ]
                    
                    match_result = self.color_matcher.match_colors(
                        [{'pad_index': pad_index, 'lab': detected_color}],
                        reference_swatches
                    )
                    
                    if match_result.get('success') and match_result.get('matches'):
                        best_match = match_result['matches'][0]
                        mapped_value = best_match.get('chemistry_value')
                        mapping_confidence = best_match.get('color_matching_confidence', 0.0)
                
                # Calculate overall confidence
                pad_confidence = pad.get('confidence', 0.7)
                overall_confidence = (pad_confidence * 0.6 + mapping_confidence * 0.4) if mapping_confidence > 0 else pad_confidence * 0.6
                
                mapped_pad: BottlePadRegion = {
                    'pad_index': pad_index,
                    'name': pad_name,
                    'region': {
                        'pad_index': pad_index,
                        'x': pad.get('x', pad.get('left', 0)),
                        'y': pad.get('y', pad.get('top', 0)),
                        'width': pad.get('width', 0),
                        'height': pad.get('height', 0),
                        'left': pad.get('left', pad.get('x', 0)),
                        'top': pad.get('top', pad.get('y', 0)),
                        'right': pad.get('right', pad.get('x', 0) + pad.get('width', 0)),
                        'bottom': pad.get('bottom', pad.get('y', 0) + pad.get('height', 0))
                    },
                    'reference_range': reference_range,
                    'reference_squares': pad_reference_squares,
                    'detected_color': detected_color,
                    'mapped_value': mapped_value,
                    'confidence': overall_confidence
                }
                
                mapped_pads.append(mapped_pad)
            
            # Calculate overall confidence
            overall_confidence = (
                sum(p['confidence'] for p in mapped_pads) / len(mapped_pads)
                if mapped_pads else 0.0
            )
            
            return {
                'success': len(mapped_pads) > 0,
                'mapped_pads': mapped_pads,
                'overall_confidence': overall_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping colors: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'COLOR_MAPPING_FAILED',
                'mapped_pads': [],
                'overall_confidence': 0.0
            }

