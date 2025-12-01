"""
Strip Detection Service - Orchestrator for YOLO-PCA pipeline.

Uses ImageTransformContext properly:
1. YOLO 1 finds strip → CROP to YOLO bbox
2. EXPAND crop (padding for PCA)
3. PCA on current image → ROTATE
4. Re-crop to expected area (transform YOLO 1 to rotated space)
5. YOLO 2 on current image → CROP to YOLO 2 bbox (if valid)
6. EXPAND crop (padding for iterative PCA)
7. PCA on current image → ROTATE (total angle)
8. Final crop and refinement
"""

import numpy as np
import cv2
import logging
from typing import Dict, Optional, Tuple

from services.utils.image_transform_context import ImageTransformContext
from services.utils.debug import DebugContext
from services.detection.yolo_detector import YoloDetector

from .context import YoloResult, DetectionContext
from .yolo_steps import run_yolo_detection
from .pca_steps import detect_rotation_with_pca
from .validation import calculate_bbox_iou, check_centering

logger = logging.getLogger(__name__)


def visualize_step(debug: DebugContext, image: np.ndarray, bbox: Optional[Dict], 
                   step_id: str, title: str, description: str, metadata: Dict = None,
                   color: Tuple[int, int, int] = (0, 255, 0)):
    """Helper to visualize a step with optional bbox overlay."""
    vis = image.copy()
    if bbox:
        cv2.rectangle(vis, (bbox.get('x1', bbox.get('left', 0)), 
                           bbox.get('y1', bbox.get('top', 0))),
                     (bbox.get('x2', bbox.get('right', 0)), 
                      bbox.get('y2', bbox.get('bottom', 0))), color, 3)
    debug.add_step(step_id, title, vis, metadata or {}, description)


class StripDetectionService:
    """
    Strip detection using YOLO-PCA pipeline with proper transform context usage.
    """
    
    # Padding constants
    PCA_PADDING = 30  # Padding around strip for PCA analysis
    YOLO2_EXPAND_PIXELS = 100  # Fixed expansion for YOLO 2 (not percentage)
    FINAL_PADDING = 20  # Final padding for output
    
    # Rotation threshold - skip rotation if angle is smaller than this
    # Small rotations degrade image quality and can hurt pad detection
    MIN_ROTATION_ANGLE = 1.0  # degrees
    
    def __init__(self, yolo_detector: YoloDetector = None, pca_config: Dict = None):
        """Initialize strip detection service."""
        self.yolo_detector = yolo_detector or YoloDetector()
        self.pca_config = pca_config or {}
        self.logger = logging.getLogger(__name__)
    
    def detect_strip(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext] = None
    ) -> Dict:
        """
        Detect test strip using YOLO-PCA pipeline.
        
        Each step uses get_current_image() from the transform context.
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context for visualization
        
        Returns:
            Dictionary with detection results and transform_context
        """
        if debug:
            debug.add_step('01_00_original', 'Original Image', image, 
                          {'shape': image.shape}, 'Starting strip detection pipeline')
        
        # Initialize transform context
        ctx = ImageTransformContext(image)
        
        # Step 1: YOLO 1 on original image
        yolo1_bbox = self._step_yolo1(ctx, debug)
        if yolo1_bbox is None:
            return self._build_failure('YOLO 1 failed to detect strip')
        
        # Step 2: Crop to YOLO 1 bbox with padding
        ctx.apply_crop(yolo1_bbox, padding=self.PCA_PADDING)
        if debug:
            visualize_step(debug, ctx.get_current_image(), None, 
                          '01_02_cropped_yolo1', 'Cropped to YOLO 1',
                          f'Cropped to YOLO 1 detection with {self.PCA_PADDING}px padding')
        
        # Step 3: PCA on cropped strip
        angle1, pca_params1 = self._step_pca(ctx, debug, '01_03_pca1', 'PCA 1')
        
        # Step 4: Apply rotation (only if above threshold)
        # Small rotations degrade image quality and can hurt pad detection
        if abs(angle1) >= self.MIN_ROTATION_ANGLE:
            ctx.apply_rotation(angle1)
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_04_rotated', 'Rotated Image',
                              f'Rotated by {angle1:.2f}° based on PCA 1')
        else:
            self.logger.info(f'Skipping rotation: angle {angle1:.2f}° below threshold {self.MIN_ROTATION_ANGLE}°')
            angle1 = 0.0  # Reset angle since we didn't apply it
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_04_rotated', 'Rotation Skipped',
                              f'Rotation skipped: {angle1:.2f}° below threshold {self.MIN_ROTATION_ANGLE}°')
        
        # Step 5: Re-crop to where strip should be (transform YOLO 1 to rotated space)
        # Add fixed expansion for YOLO 2 to have room to find the strip
        expected_bbox = ctx.transform_coords_original_to_rotated(yolo1_bbox)
        ctx.apply_crop(expected_bbox, padding=self.YOLO2_EXPAND_PIXELS)
        if debug:
            visualize_step(debug, ctx.get_current_image(), None,
                          '01_05_cropped_for_yolo2', 'Cropped for YOLO 2',
                          f'Cropped to expected area with {self.YOLO2_EXPAND_PIXELS}px expansion')
        
        # Step 6: YOLO 2 on cropped rotated image
        yolo2_bbox = self._step_yolo2(ctx, debug)
        if yolo2_bbox is None:
            return self._build_failure('YOLO 2 failed to detect strip')
        
        # Step 7: Crop to YOLO 2 bbox with padding for iterative PCA
        # yolo2_bbox is in cropped space, transform to rotated space
        yolo2_in_rotated = ctx.transform_coords_to_rotated(yolo2_bbox)
        ctx.apply_crop(yolo2_in_rotated, padding=self.PCA_PADDING)
        if debug:
            visualize_step(debug, ctx.get_current_image(), None,
                          '01_07_cropped_yolo2', 'Cropped to YOLO 2',
                          f'Cropped to YOLO 2 detection with {self.PCA_PADDING}px padding')
        
        # Step 8: Iterative PCA (on current cropped image)
        angle2, pca_params2 = self._step_pca(ctx, debug, '01_08_pca2', 'PCA 2 (Iterative)')
        total_angle = angle1 + angle2
        
        # Step 9: Apply total rotation (only if above threshold)
        # Small rotations degrade image quality and can hurt pad detection
        if abs(total_angle) >= self.MIN_ROTATION_ANGLE:
            ctx.apply_rotation(total_angle)
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_09_rotated_final', 'Final Rotation',
                              f'Total rotation: {total_angle:.2f}° ({angle1:.2f}° + {angle2:.2f}°)')
        else:
            self.logger.info(f'Skipping final rotation: total angle {total_angle:.2f}° below threshold {self.MIN_ROTATION_ANGLE}°')
            total_angle = 0.0  # Reset since we didn't apply
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_09_rotated_final', 'Final Rotation Skipped',
                              f'Rotation skipped: {total_angle:.2f}° below threshold {self.MIN_ROTATION_ANGLE}°')
        
        # Step 10: Final crop - transform YOLO 1 to new rotated space
        final_expected = ctx.transform_coords_original_to_rotated(yolo1_bbox)
        ctx.apply_crop(final_expected, padding=self.FINAL_PADDING)
        if debug:
            visualize_step(debug, ctx.get_current_image(), None,
                          '01_10_final_crop', 'Final Strip Crop',
                          f'Final cropped strip with {self.FINAL_PADDING}px padding')
        
        # Build result
        ctx.set_metadata('yolo_pca_iterative', 1.0)
        
        return {
            'success': True,
            'strip_region': self._build_strip_region(ctx, yolo1_bbox),
            'transform_context': ctx,
            'detection_method': 'yolo_pca_iterative',
            'rotation_angle': total_angle,
            'yolo1_bbox': yolo1_bbox,
            'confidence': 1.0
        }
    
    def _step_yolo1(self, ctx: ImageTransformContext, debug: Optional[DebugContext]) -> Optional[Dict]:
        """Step 1: Run YOLO on original image."""
        result = run_yolo_detection(self.yolo_detector, ctx.get_current_image(), "yolo1")
        
        if not result.success:
            self.logger.error('YOLO 1 failed to detect strip')
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_01_yolo1_failed', 'YOLO 1 Failed',
                              'YOLO failed to detect test strip', color=(0, 0, 255))
            return None
        
        if debug:
            visualize_step(debug, ctx.get_current_image(), result.bbox,
                          '01_01_yolo1', 'YOLO 1 Detection',
                          f'YOLO detected strip with confidence {result.confidence:.3f}',
                          {'confidence': result.confidence, 'bbox': result.bbox})
        
        return result.bbox
    
    def _step_pca(self, ctx: ImageTransformContext, debug: Optional[DebugContext],
                  step_id: str, title: str) -> Tuple[float, Dict]:
        """Run PCA rotation detection on current image."""
        current_img = ctx.get_current_image()
        h, w = current_img.shape[:2]
        
        # Create a bbox covering the full current image for PCA
        full_bbox = {'x1': 0, 'y1': 0, 'x2': w, 'y2': h}
        
        angle, pca_params = detect_rotation_with_pca(current_img, full_bbox, self.pca_config)
        
        if debug:
            vis = current_img.copy()
            cv2.putText(vis, f'Angle: {angle:.2f}°', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug.add_step(step_id, title, vis,
                          {'angle': angle, 'pca_params': pca_params},
                          f'PCA detected rotation angle: {angle:.2f}°')
        
        return angle, pca_params
    
    def _step_yolo2(self, ctx: ImageTransformContext, debug: Optional[DebugContext]) -> Optional[Dict]:
        """Step 6: Run YOLO 2 on current (cropped rotated) image."""
        result = run_yolo_detection(self.yolo_detector, ctx.get_current_image(), "yolo2")
        
        if not result.success:
            self.logger.error('YOLO 2 failed to detect strip')
            if debug:
                visualize_step(debug, ctx.get_current_image(), None,
                              '01_06_yolo2_failed', 'YOLO 2 Failed',
                              'YOLO 2 failed to detect strip in cropped area', color=(0, 0, 255))
            return None
        
        # Validate that detection is reasonably centered (strip should be near center of cropped area)
        h, w = ctx.get_current_shape()
        if not check_centering(result.bbox, (h, w), margin=0.3):
            self.logger.warning('YOLO 2 detection not centered, may be incorrect')
            if debug:
                visualize_step(debug, ctx.get_current_image(), result.bbox,
                              '01_06_yolo2_offcenter', 'YOLO 2 Off-Center',
                              f'YOLO 2 detection not centered - confidence {result.confidence:.3f}',
                              color=(0, 165, 255))
            # Still use it but log warning
        
        if debug:
            visualize_step(debug, ctx.get_current_image(), result.bbox,
                          '01_06_yolo2', 'YOLO 2 Detection',
                          f'YOLO 2 detected strip with confidence {result.confidence:.3f}',
                          {'confidence': result.confidence, 'bbox': result.bbox})
        
        return result.bbox
    
    def _build_strip_region(self, ctx: ImageTransformContext, yolo1_bbox: Dict) -> Dict:
        """Build the strip region dictionary from context."""
        current = ctx.get_current_image()
        h, w = current.shape[:2]
        
        return {
            'x': 0,
            'y': 0,
            'width': w,
            'height': h,
            'x1': 0,
            'y1': 0,
            'x2': w,
            'y2': h,
            'left': 0,
            'top': 0,
            'right': w,
            'bottom': h,
            'rotation_angle': ctx.rotation_angle,
            'original_bbox': yolo1_bbox
        }
    
    def _build_failure(self, error: str) -> Dict:
        """Build a failure result."""
        return {
            'success': False,
            'error': error,
            'strip_region': None,
            'transform_context': None
        }
