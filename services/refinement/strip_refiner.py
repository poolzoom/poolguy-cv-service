"""
Main strip refinement orchestrator.

Coordinates all refinement steps: orientation → projection → edges → tightening → padding.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional
from services.interfaces import StripRegion
from services.utils.debug import DebugContext
from config.refinement_config import REFINEMENT_CONFIG, get_config

from .methods.intensity_projector import IntensityProjector
from .methods.bbox_tightener import BboxTightener
from .methods.strip_preparator import StripPreparator

logger = logging.getLogger(__name__)


class StripRefiner:
    """
    Main orchestrator for strip refinement pipeline.
    
    Refines YOLO-detected strip regions through multiple stages:
    1. Intensity projection (finds left/right and top/bottom boundaries)
    2. Bounding box tightening (applies boundaries)
    
    Note: Image is expected to already be rotated from yolo_pca pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize strip refiner.
        
        Args:
            config: Refinement configuration dict (uses default if None)
        """
        self.config = config or get_config()
        
        # Initialize refinement methods
        self.intensity_projector = IntensityProjector(self.config.get('projection', {}))
        self.bbox_tightener = BboxTightener(self.config.get('tightening', {}))
        self.strip_preparator = StripPreparator(self.config.get('preparation', {}))
    
    def refine(
        self,
        image: np.ndarray,
        yolo_bbox: Dict,
        debug: Optional[DebugContext] = None
    ) -> StripRegion:
        """
        Refine YOLO-detected strip region.
        
        Args:
            image: Original full image (BGR format)
            yolo_bbox: YOLO bounding box dict with 'x1', 'y1', 'x2', 'y2'
            debug: Optional debug context for visual logging
            
        Returns:
            Refined StripRegion with absolute coordinates
        """
        # Convert YOLO bbox to region dict
        region = {
            'left': yolo_bbox['x1'],
            'top': yolo_bbox['y1'],
            'right': yolo_bbox['x2'],
            'bottom': yolo_bbox['y2']
        }
        
        current_image = image.copy()
        current_region = region.copy()
        
        if debug:
            vis = image.copy()
            cv2.rectangle(vis, (region['left'], region['top']),
                         (region['right'], region['bottom']), (255, 0, 0), 3)  # Blue for YOLO
            cv2.putText(vis, 'YOLO Detection', (region['left'], region['top'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            debug.add_step('refinement_00_yolo_input', 'Refinement: YOLO Detection Input', vis, {'bbox': yolo_bbox})
        
        # Image is already rotated from yolo_pca pipeline, so we work directly with it
        # Extract the cropped region
        x1, y1 = current_region['left'], current_region['top']
        x2, y2 = current_region['right'], current_region['bottom']
        working_image = current_image[y1:y2, x1:x2].copy()
        crop_offset_x = x1
        crop_offset_y = y1
        
        # Create a region dict relative to the cropped image
        crop_region = {
            'left': 0,
            'top': 0,
            'right': working_image.shape[1],
            'bottom': working_image.shape[0]
        }
        
        # Step A: Project intensities (find left/right AND top/bottom boundaries)
        try:
            crop_region, projection_meta = self.intensity_projector.refine(
                working_image, crop_region, debug, yolo_constraint=None
            )
        except Exception as e:
            logger.warning(f"Intensity projection failed: {e}")
        
        # Step B: Tighten bounding box (apply boundaries from projection)
        try:
            crop_region, tightening_meta = self.bbox_tightener.refine(
                working_image, crop_region, debug
            )
        except Exception as e:
            logger.warning(f"Bounding box tightening failed: {e}")
        
        # Step C: Prepare strip for detection (color-safe processing)
        prepared_image = None
        try:
            # Create absolute region for preparator (it needs to crop from working_image)
            abs_region_for_prep = {
                'left': crop_region['left'],
                'top': crop_region['top'],
                'right': crop_region['right'],
                'bottom': crop_region['bottom']
            }
            _, prep_meta = self.strip_preparator.refine(
                working_image, abs_region_for_prep, debug
            )
            prepared_image = prep_meta.get('prepared_image')
        except Exception as e:
            logger.warning(f"Strip preparation failed: {e}")
        
        # Convert crop_region back to absolute coordinates
        # Simple offset addition (no rotation transform needed since rotation already done)
        current_region = {
            'left': crop_region['left'] + crop_offset_x,
            'top': crop_region['top'] + crop_offset_y,
            'right': crop_region['right'] + crop_offset_x,
            'bottom': crop_region['bottom'] + crop_offset_y
        }
        
        # Convert to StripRegion format
        width = current_region['right'] - current_region['left']
        height = current_region['bottom'] - current_region['top']
        orientation = 'vertical' if height > width else 'horizontal'
        
        refined_strip = StripRegion(
            left=current_region['left'],
            top=current_region['top'],
            right=current_region['right'],
            bottom=current_region['bottom'],
            width=width,
            height=height,
            confidence=1.0,  # Refinement confidence (could be calculated)
            detection_method='yolo+refined',
            orientation=orientation,
            angle=0.0  # Rotation already applied in yolo_pca pipeline
        )
        
        # Store prepared image in the strip region dict for pipeline to use
        if prepared_image is not None:
            refined_strip['prepared_image'] = prepared_image
            refined_strip['preparation_scale'] = prep_meta.get('scale', 1.0)
        
        if debug:
            vis = image.copy()
            cv2.rectangle(vis, (region['left'], region['top']),
                         (region['right'], region['bottom']), (255, 0, 0), 2)  # Original YOLO
            cv2.rectangle(vis, (refined_strip['left'], refined_strip['top']),
                         (refined_strip['right'], refined_strip['bottom']), (0, 255, 0), 3)  # Refined
            cv2.putText(vis, 'Refined Strip', 
                       (refined_strip['left'], refined_strip['top'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug.add_step('refinement_99_final', 'Refinement: Final Refined Strip', vis, {
                'original_bbox': region,
                'refined_bbox': current_region
            })
        
        return refined_strip

