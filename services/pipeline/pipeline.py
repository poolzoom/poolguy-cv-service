"""
Pipeline service for processing test strip images.

Orchestrates: Strip Detection → Pad Detection → Color Extraction
"""

import logging
import numpy as np
import cv2
from typing import Dict, Optional

from services.pipeline.steps.strip_detection import StripDetectionService
from services.pipeline.steps.pad_detection import PadDetectionService
from services.pipeline.steps.color_extraction import ColorExtractionService
from services.interfaces import StripRegion
from services.utils.debug import DebugContext
from services.utils.image_transform_context import ImageTransformContext
from utils.coordinate_transform import transform_pad_coordinates_to_absolute

logger = logging.getLogger(__name__)


class PipelineService:
    """
    Main pipeline service that processes test strip images.
    
    Pipeline: Image → Strip Detection → Pad Detection → Color Extraction
    """
    
    def __init__(
        self,
        strip_detection_service: Optional[StripDetectionService] = None,
        pad_detection_service: Optional[PadDetectionService] = None,
        color_extraction_service: Optional[ColorExtractionService] = None
    ):
        """
        Initialize pipeline service.
        
        Args:
            strip_detection_service: Optional pre-initialized service
            pad_detection_service: Optional pre-initialized service
            color_extraction_service: Optional pre-initialized service
        """
        # Initialize strip detection service
        self.strip_detection_service = strip_detection_service or StripDetectionService()
        self.pad_detection_service = pad_detection_service or PadDetectionService()
        self.color_extraction_service = color_extraction_service or ColorExtractionService()
    
    def process_image(
        self,
        image: np.ndarray,
        image_name: str = "unknown",
        expected_pad_count: int = 6,
        normalize_white: bool = True,
        debug: Optional[DebugContext] = None
    ) -> Dict:
        """
        Process image through full pipeline.
        
        Args:
            image: Input image (BGR format)
            image_name: Name of image for logging
            expected_pad_count: Expected number of pads (4-7)
            normalize_white: Whether to apply white balance normalization
            debug: Optional DebugContext for visual logging
        
        Returns:
            Result dictionary with success status and data or error
        """
        # Step 1: Detect strip
        strip_result = self.strip_detection_service.detect_strip(image=image, debug=debug)
        
        if not strip_result.get('success'):
            if debug:
                vis_error = debug.visualize_error(
                    image,
                    strip_result.get('error', 'Strip detection failed'),
                    'STRIP_DETECTION_FAILED'
                )
                debug.add_step('00_strip_detection_failed', 'Strip Detection Failed', vis_error, {
                    'error': strip_result.get('error', 'Unknown error'),
                    'error_code': 'STRIP_DETECTION_FAILED'
                })
            return {
                'success': False,
                'error': strip_result.get('error', 'Strip detection failed'),
                'error_code': 'STRIP_DETECTION_FAILED'
            }
        
        # Debug: Step 01 - Strip detection result
        if debug:
            strip_region_dict = strip_result['strip_region']
            vis_strip = debug.visualize_strip_detection(
                image,
                strip_region_dict,
                strip_result.get('method', 'unknown'),
                strip_result.get('confidence', 0.0)
            )
            method = strip_result.get('method', 'unknown')
            confidence = strip_result.get('confidence', 0.0)
            angle = strip_result.get('angle', 0.0)
            debug.add_step(
                '00_strip_detection_result', 
                'Strip Detection Result', 
                vis_strip, 
                {
                    'method': method,
                    'confidence': confidence,
                    'strip_region': strip_region_dict,
                    'orientation': strip_result.get('orientation', 'vertical'),
                    'angle': angle
                },
                f'Strip detected using {method} method with confidence {confidence:.3f}, rotation angle: {angle:.2f}°'
            )
        
        # Step 2: Get transform context from strip detection
        transform_context: Optional[ImageTransformContext] = strip_result.get('transform_context')
        strip_region_dict = strip_result['strip_region']
        strip_region: StripRegion = {
            'left': strip_region_dict['left'],
            'top': strip_region_dict['top'],
            'right': strip_region_dict['right'],
            'bottom': strip_region_dict['bottom'],
            'width': strip_region_dict.get('width', strip_region_dict['right'] - strip_region_dict['left']),
            'height': strip_region_dict.get('height', strip_region_dict['bottom'] - strip_region_dict['top']),
            'confidence': strip_region_dict.get('confidence', 0.0),
            'detection_method': strip_result.get('method', 'unknown'),
            'orientation': strip_result.get('orientation', 'vertical'),
            'angle': strip_result.get('angle', 0.0)
        }
        
        # Step 3: Get working image from context (rotated and cropped strip)
        if transform_context is None:
            # Fallback: create minimal context if not provided
            transform_context = ImageTransformContext(image)
            # Crop from original image as fallback
            try:
                transform_context.apply_crop(strip_region)
            except Exception as e:
                if debug:
                    vis_error = debug.visualize_error(
                        image,
                        f'Failed to crop strip region: {str(e)}',
                        'STRIP_CROP_FAILED'
                    )
                    debug.add_step('00_strip_crop_failed', 'Strip Crop Failed', vis_error, {
                        'error': str(e),
                        'error_code': 'STRIP_CROP_FAILED'
                    })
                return {
                    'success': False,
                    'error': f'Failed to crop strip region: {str(e)}',
                    'error_code': 'STRIP_CROP_FAILED'
                }
        
        # Get working image (cropped > rotated > original)
        working_image = transform_context.get_current_image()
        
        # Debug: Step 02 - Working strip image
        if debug:
            w, h = working_image.shape[1], working_image.shape[0]
            has_cropped = transform_context.cropped_image is not None
            rotation_angle = transform_context.rotation_angle
            desc = f'Working strip image: {w}x{h}px'
            if has_cropped:
                desc += ', cropped'
            if rotation_angle != 0:
                desc += f', rotated {rotation_angle:.2f}°'
            debug.add_step(
                '00_strip_crop_result', 
                'Working Strip Image', 
                working_image, 
                {
                    'width': w,
                    'height': h,
                    'has_cropped': has_cropped,
                    'rotation_applied': transform_context.rotation_applied,
                    'rotation_angle': rotation_angle,
                    'crop_offset': transform_context.crop_offset,
                    'crop_shape': transform_context.crop_shape,
                    'strip_region_original': {
                        'left': strip_region['left'],
                        'top': strip_region['top'],
                        'right': strip_region['right'],
                        'bottom': strip_region['bottom']
                    }
                },
                desc
            )
        
        # Step 4: Detect pads in working image
        pad_detection_result = self.pad_detection_service.detect_pads_in_strip(
            strip_image=working_image,
            strip_region=strip_region,
            expected_pad_count=expected_pad_count,
            debug=debug
        )
        
        # Continue processing even if pad detection "failed" - use whatever pads were detected
        detected_pads = pad_detection_result.get('pads', [])
        detected_count = pad_detection_result.get('detected_count', 0)
        
        if not pad_detection_result['success']:
            # Log warning but continue with detected pads (if any)
            logger.warning(f'Pad detection reported failure: {pad_detection_result.get("error")}, but continuing with {detected_count} detected pads')
            
            if debug:
                # Show detected pads for debugging
                vis_error = working_image.copy()
                
                # Draw any pads that were detected
                for i, pad in enumerate(detected_pads):
                    x = pad.get('x', pad.get('left', 0))
                    y = pad.get('y', pad.get('top', 0))
                    width = pad.get('width', 0)
                    height = pad.get('height', 0)
                    conf = pad.get('confidence', 0.0)
                    cv2.rectangle(vis_error, (x, y), (x + width, y + height), (0, 165, 255), 2)  # Orange for warning
                    cv2.putText(vis_error, f'P{i+1} {conf:.2f}', (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
                # Add warning text
                error_msg = pad_detection_result.get('error', 'Pad detection failed')
                cv2.putText(vis_error, f'WARNING: {error_msg}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(vis_error, f'Detected: {detected_count}, Expected: {expected_pad_count}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                debug.add_step('00_pad_detection_warning', 'Pad Detection Warning', vis_error, {
                    'error': pad_detection_result.get('error', 'Unknown error'),
                    'error_code': pad_detection_result.get('error_code', 'PAD_DETECTION_FAILED'),
                    'detected_count': detected_count,
                    'expected_count': expected_pad_count,
                    'pads': [
                        {
                            'pad_index': i,
                            'x': pad.get('x', pad.get('left', 0)),
                            'y': pad.get('y', pad.get('top', 0)),
                            'width': pad.get('width', 0),
                            'height': pad.get('height', 0),
                            'confidence': pad.get('confidence', 0.0)
                        }
                        for i, pad in enumerate(detected_pads)
                    ]
                })
            
            # If no pads detected at all, then fail
            if detected_count == 0:
                return {
                    'success': False,
                    'error': pad_detection_result.get('error', 'Pad detection failed - no pads detected'),
                    'error_code': pad_detection_result.get('error_code', 'PAD_DETECTION_FAILED')
                }
        
        # Log warning if pad count doesn't match, but continue processing
        if detected_count != expected_pad_count:
            logger.warning(
                f'Pad count mismatch: expected {expected_pad_count}, detected {detected_count}. '
                f'Continuing with detected pads.'
            )
        
        # Step 5: Transform pad coordinates from working space to original space
        relative_pads = pad_detection_result['pads']
        
        # Transform pad coordinates from working (cropped) space to original image space
        absolute_pads = []
        for pad in relative_pads:
            # Transform from working space to original using context
            original_coords = transform_context.transform_coords_to_original(pad)
            # Convert to PadRegion format
            from services.interfaces import PadRegion
            absolute_pad = PadRegion(
                pad_index=pad.get('pad_index', 0),
                x=original_coords.get('x', original_coords.get('left', 0)),
                y=original_coords.get('y', original_coords.get('top', 0)),
                width=original_coords.get('width', 0),
                height=original_coords.get('height', 0),
                left=original_coords.get('left', original_coords.get('x', 0)),
                top=original_coords.get('top', original_coords.get('y', 0)),
                right=original_coords.get('right', original_coords.get('x', 0) + original_coords.get('width', 0)),
                bottom=original_coords.get('bottom', original_coords.get('y', 0) + original_coords.get('height', 0))
            )
            absolute_pads.append(absolute_pad)
        
        # Debug: Step 03 - Pad detection result (show both relative and absolute)
        if debug:
            # Visualize on the working image (cropped)
            vis_pads_relative = debug.visualize_pad_detection(
                working_image,
                relative_pads,
                expected_pad_count
            )
            detected_count = pad_detection_result['detected_count']
            debug.add_step(
                '00_pad_detection_relative', 
                'Pad Detection (Working Space)', 
                vis_pads_relative, 
                {
                    'detected_count': detected_count,
                    'expected_count': expected_pad_count,
                    'coordinate_system': 'working_space',
                    'pads': [
                        {
                            'pad_index': i,
                            'x': pad.get('x', pad.get('left', 0)),
                            'y': pad.get('y', pad.get('top', 0)),
                            'width': pad.get('width', 0),
                            'height': pad.get('height', 0)
                        }
                        for i, pad in enumerate(relative_pads)
                    ]
                },
                f'Detected {detected_count} pads in working space (expected {expected_pad_count})'
            )
            
            # Note: Removed 00_pad_detection_absolute - it was misleading because
            # it showed axis-aligned boxes on the original image where the strip is rotated.
            # All operations (pad detection, color extraction) work on working_image.
        
        # Step 6: Extract colors using current working image
        color_extraction_image = working_image
        # relative_pads are already in cropped_strip coordinates (scaled back from prepared if needed)
        result = self.color_extraction_service.extract_colors(
            image=color_extraction_image,
            pad_regions=relative_pads,
            expected_pad_count=expected_pad_count,
            normalize_white=normalize_white,
            debug=debug
        )
        
        if not result.get('success'):
            return {
                'success': False,
                'error': result.get('error', 'Color extraction failed'),
                'error_code': result.get('error_code', 'COLOR_EXTRACTION_FAILED')
            }
        
        # Update pad regions with absolute coordinates
        pads_data = result.get('data', {}).get('pads', [])
        for i, pad in enumerate(pads_data):
            if i < len(absolute_pads):
                pad['region'] = absolute_pads[i]
        
        # Add pad count mismatch info to result if applicable
        detected_count = pad_detection_result['detected_count']
        if detected_count != expected_pad_count:
            result['data']['pad_count_mismatch'] = {
                'expected': expected_pad_count,
                'detected': detected_count,
                'warning': f'Expected {expected_pad_count} pads, detected {detected_count}'
            }
        
        # Debug: Step 04 - Final result with pad colors (on working image)
        if debug:
            overall_confidence = result.get('data', {}).get('overall_confidence', 0.0)
            
            # Create pads_data with relative coords for visualization on working_image
            pads_data_relative = []
            for i, pad in enumerate(pads_data):
                pad_relative = pad.copy()
                if i < len(relative_pads):
                    # Use relative coords for visualization
                    pad_relative['region'] = {
                        'x': relative_pads[i].get('x', relative_pads[i].get('left', 0)),
                        'y': relative_pads[i].get('y', relative_pads[i].get('top', 0)),
                        'width': relative_pads[i].get('width', 0),
                        'height': relative_pads[i].get('height', 0)
                    }
                pads_data_relative.append(pad_relative)
            
            # Working strip region (full image since it's already cropped)
            working_h, working_w = working_image.shape[:2]
            working_strip_region = {
                'left': 0, 'top': 0, 'right': working_w, 'bottom': working_h
            }
            
            vis_final = debug.visualize_final_result(
                working_image,
                working_strip_region,
                pads_data_relative,
                overall_confidence
            )
            debug_data = {
                'pad_count': len(pads_data),
                'expected_count': expected_pad_count,
                'detected_count': detected_count,
                'overall_confidence': overall_confidence,
                'coordinate_system': 'working_space',
                'pads': [
                    {
                        'pad_index': i,
                        'lab': pad.get('lab', {}),
                        'confidence': pad.get('confidence', 0.0),
                        'region': pad.get('region', {})
                    }
                    for i, pad in enumerate(pads_data_relative)
                ]
            }
            if detected_count != expected_pad_count:
                debug_data['pad_count_mismatch'] = {
                    'expected': expected_pad_count,
                    'detected': detected_count
                }
            overall_conf = overall_confidence
            pad_count = len(pads_data)
            desc = f'Final result: {pad_count} pads detected, overall confidence: {overall_conf:.3f}'
            if detected_count != expected_pad_count:
                desc += f' (expected {expected_pad_count})'
            debug.add_step('00_final_result', 'Final Result', vis_final, debug_data, desc)
        
        return result

