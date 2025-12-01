"""
Multi-image handler for bottle pipeline.
Handles multiple images when pads wrap around the bottle, detecting overlaps and merging results.

DEPRECATED: This service is no longer used in the new stateless API.
The new API processes one image at a time - Laravel handles image management.
Kept for backwards compatibility only.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from services.interfaces import BottlePadRegion

logger = logging.getLogger(__name__)


class MultiImageHandler:
    """
    Service for handling multiple images of the same bottle.
    
    Detects overlapping pads across images and merges results.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize multi-image handler.
        
        Args:
            iou_threshold: IoU threshold for considering pads as duplicates (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.iou_threshold = iou_threshold
    
    def merge_pad_results(
        self,
        image_results: List[Dict],
        debug: Optional[object] = None
    ) -> Dict:
        """
        Merge pad detection results from multiple images.
        
        Args:
            image_results: List of results from each image:
                [
                    {
                        'image_index': int,
                        'pads': List[BottlePadRegion],
                        'reference_squares': List[ReferenceSquare],
                        ...
                    },
                    ...
                ]
            debug: Optional debug context
        
        Returns:
            Dictionary with merged results:
            {
                'success': bool,
                'merged_pads': List[BottlePadRegion],
                'merged_reference_squares': List[ReferenceSquare],
                'images_processed': int,
                'pads_detected': int,
                'overlaps_detected': int
            }
        """
        if not image_results:
            return {
                'success': False,
                'error': 'No image results provided',
                'error_code': 'INVALID_PARAMETER',
                'merged_pads': [],
                'merged_reference_squares': [],
                'images_processed': 0,
                'pads_detected': 0,
                'overlaps_detected': 0
            }
        
        # Collect all pads from all images
        all_pads = []
        all_reference_squares = []
        
        for img_result in image_results:
            image_index = img_result.get('image_index', 0)
            pads = img_result.get('pads', [])
            ref_squares = img_result.get('reference_squares', [])
            
            # Tag pads with image index
            for pad in pads:
                pad['_image_index'] = image_index
                all_pads.append(pad)
            
            # Tag reference squares with image index
            for ref_square in ref_squares:
                ref_square['_image_index'] = image_index
                all_reference_squares.append(ref_square)
        
        # Detect and merge overlapping pads
        merged_pads = self._merge_overlapping_pads(all_pads)
        
        # Merge reference squares (keep unique ones)
        merged_reference_squares = self._merge_reference_squares(all_reference_squares)
        
        # Count overlaps
        overlaps_detected = len(all_pads) - len(merged_pads)
        
        return {
            'success': True,
            'merged_pads': merged_pads,
            'merged_reference_squares': merged_reference_squares,
            'images_processed': len(image_results),
            'pads_detected': len(merged_pads),
            'overlaps_detected': overlaps_detected
        }
    
    def _merge_overlapping_pads(self, pads: List[Dict]) -> List[BottlePadRegion]:
        """
        Merge overlapping pads using IoU (Intersection over Union).
        
        Args:
            pads: List of pad regions from all images
        
        Returns:
            List of merged pad regions
        """
        if not pads:
            return []
        
        # Group pads by IoU similarity
        merged = []
        used = set()
        
        for i, pad1 in enumerate(pads):
            if i in used:
                continue
            
            # Find all overlapping pads
            overlapping = [pad1]
            used.add(i)
            
            for j, pad2 in enumerate(pads[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(pad1, pad2)
                if iou >= self.iou_threshold:
                    overlapping.append(pad2)
                    used.add(j)
            
            # Merge overlapping pads
            merged_pad = self._merge_pad_group(overlapping)
            merged.append(merged_pad)
        
        # Sort by pad_index
        merged.sort(key=lambda p: p.get('pad_index', 0))
        
        # Reassign pad_index sequentially
        for i, pad in enumerate(merged):
            pad['pad_index'] = i
            if pad.get('region'):
                pad['region']['pad_index'] = i
        
        return merged
    
    def _calculate_iou(self, pad1: Dict, pad2: Dict) -> float:
        """
        Calculate Intersection over Union (IoU) between two pads.
        
        Args:
            pad1: First pad region
            pad2: Second pad region
        
        Returns:
            IoU value (0.0-1.0)
        """
        # Get coordinates
        def get_coords(pad):
            region = pad.get('region', pad)
            return {
                'left': region.get('left', region.get('x', 0)),
                'top': region.get('top', region.get('y', 0)),
                'right': region.get('right', region.get('x', 0) + region.get('width', 0)),
                'bottom': region.get('bottom', region.get('y', 0) + region.get('height', 0))
            }
        
        coords1 = get_coords(pad1)
        coords2 = get_coords(pad2)
        
        # Calculate intersection
        left = max(coords1['left'], coords2['left'])
        top = max(coords1['top'], coords2['top'])
        right = min(coords1['right'], coords2['right'])
        bottom = min(coords1['bottom'], coords2['bottom'])
        
        if right <= left or bottom <= top:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        
        # Calculate union
        area1 = (coords1['right'] - coords1['left']) * (coords1['bottom'] - coords1['top'])
        area2 = (coords2['right'] - coords2['left']) * (coords2['bottom'] - coords2['top'])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _merge_pad_group(self, pads: List[Dict]) -> BottlePadRegion:
        """
        Merge a group of overlapping pads into a single pad.
        
        Args:
            pads: List of overlapping pad regions
        
        Returns:
            Merged pad region
        """
        if len(pads) == 1:
            # Remove internal tags
            pad = pads[0].copy()
            pad.pop('_image_index', None)
            return pad
        
        # Use the pad with highest confidence as base
        best_pad = max(pads, key=lambda p: p.get('confidence', 0.0))
        
        # Merge coordinates (use bounding box of all pads)
        all_left = []
        all_top = []
        all_right = []
        all_bottom = []
        
        for pad in pads:
            region = pad.get('region', pad)
            all_left.append(region.get('left', region.get('x', 0)))
            all_top.append(region.get('top', region.get('y', 0)))
            all_right.append(region.get('right', region.get('x', 0) + region.get('width', 0)))
            all_bottom.append(region.get('bottom', region.get('y', 0) + region.get('height', 0)))
        
        merged_left = min(all_left)
        merged_top = min(all_top)
        merged_right = max(all_right)
        merged_bottom = max(all_bottom)
        
        # Merge other attributes (prefer non-None values)
        merged_name = next((p.get('name') for p in pads if p.get('name')), None)
        merged_reference_range = next((p.get('reference_range') for p in pads if p.get('reference_range')), None)
        merged_detected_color = next((p.get('detected_color') for p in pads if p.get('detected_color')), None)
        merged_mapped_value = next((p.get('mapped_value') for p in pads if p.get('mapped_value')), None)
        
        # Merge reference squares (deduplicate)
        all_ref_squares = []
        seen_colors = set()
        for pad in pads:
            ref_squares = pad.get('reference_squares', [])
            for ref_square in ref_squares:
                color_key = (
                    ref_square.get('color', {}).get('L', 0),
                    ref_square.get('color', {}).get('a', 0),
                    ref_square.get('color', {}).get('b', 0)
                )
                if color_key not in seen_colors:
                    all_ref_squares.append(ref_square)
                    seen_colors.add(color_key)
        
        # Calculate average confidence
        avg_confidence = sum(p.get('confidence', 0.0) for p in pads) / len(pads)
        
        merged_pad: BottlePadRegion = {
            'pad_index': best_pad.get('pad_index', 0),
            'name': merged_name,
            'region': {
                'pad_index': best_pad.get('pad_index', 0),
                'x': merged_left,
                'y': merged_top,
                'width': merged_right - merged_left,
                'height': merged_bottom - merged_top,
                'left': merged_left,
                'top': merged_top,
                'right': merged_right,
                'bottom': merged_bottom
            },
            'reference_range': merged_reference_range,
            'reference_squares': all_ref_squares,
            'detected_color': merged_detected_color,
            'mapped_value': merged_mapped_value,
            'confidence': avg_confidence
        }
        
        return merged_pad
    
    def _merge_reference_squares(self, ref_squares: List[Dict]) -> List[Dict]:
        """
        Merge reference squares, removing duplicates.
        
        Args:
            ref_squares: List of reference squares from all images
        
        Returns:
            List of unique reference squares
        """
        if not ref_squares:
            return []
        
        # Deduplicate by color similarity
        unique_squares = []
        seen_colors = set()
        
        for ref_square in ref_squares:
            color = ref_square.get('color', {})
            color_key = (
                round(color.get('L', 0), 1),
                round(color.get('a', 0), 1),
                round(color.get('b', 0), 1)
            )
            
            if color_key not in seen_colors:
                # Remove internal tags
                square = ref_square.copy()
                square.pop('_image_index', None)
                unique_squares.append(square)
                seen_colors.add(color_key)
        
        return unique_squares



