"""
Strip Detection Service for PoolGuy CV Service.

Provides multiple detection methods:
- yolo_pca: YOLO → PCA rotation → Rotate → YOLO (new simplified pipeline)
- yolo_refined: YOLO + refinement pipeline (existing)
- yolo: YOLO only (no refinement)
- openai: OpenAI vision API
- auto: Try methods in order until one succeeds
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Literal
from pathlib import Path

from services.detection.yolo_detector import YoloDetector
from services.refinement.strip_refiner import StripRefiner
from services.detection.openai_vision import OpenAIVisionService
from services.interfaces import StripRegion
from services.utils.debug import DebugContext
from services.detection.detector_adapters import yolo_result_to_strip_region, openai_result_to_strip_region
from config.pca_config import get_pca_config
from services.utils.image_transform_context import ImageTransformContext

# Optional sklearn for PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. PCA rotation detection will not work. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


def calculate_bbox_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bbox with 'x1', 'y1', 'x2', 'y2'
        bbox2: Second bbox with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        IoU value between 0 and 1
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1['x1'], bbox1['y1'], bbox1['x2'], bbox1['y2']
    x1_2, y1_2, x2_2, y2_2 = bbox2['x1'], bbox2['y1'], bbox2['x2'], bbox2['y2']
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0  # No intersection
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def detect_rotation_with_pca(
    image: np.ndarray,
    bbox: Dict,
    config: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Detect rotation angle using PCA on foreground pixels.
    
    Args:
        image: Full image (BGR format)
        bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2'
        config: Optional PCA configuration dict
    
    Returns:
        Tuple of (rotation_angle, pca_params):
        - rotation_angle: Rotation angle in degrees (positive = counterclockwise)
        - pca_params: Dictionary with PCA analysis details
    """
    if not SKLEARN_AVAILABLE:
        return 0.0, {'error': 'sklearn not available'}
    
    pca_cfg = config or get_pca_config()
    expand_x_ratio = pca_cfg.get('horizontal_expansion', 0.10)
    expand_y_ratio = pca_cfg.get('vertical_expansion', 0.05)
    min_points = pca_cfg.get('min_foreground_points', 100)
    
    # Crop to bbox region
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    # Expand bbox to include edges that might be slightly outside YOLO detection
    h, w = image.shape[:2]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    expand_x = int(bbox_width * expand_x_ratio)
    expand_y = int(bbox_height * expand_y_ratio)
    
    x1 = max(0, x1 - expand_x)
    y1 = max(0, y1 - expand_y)
    x2 = min(w, x2 + expand_x)
    y2 = min(h, y2 + expand_y)
    
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, {'error': 'Empty crop', 'crop_size': (0, 0)}
    
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get foreground pixels (the strip itself)
    # Use Otsu's method for automatic threshold selection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Get all foreground pixel coordinates (where binary > 0)
    foreground_points = np.column_stack(np.where(binary > 0))
    
    threshold_method = 'otsu'
    if len(foreground_points) < min_points:
        # Not enough foreground pixels, try adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        foreground_points = np.column_stack(np.where(binary > 0))
        threshold_method = 'adaptive'
    
    if len(foreground_points) < min_points:
        # Still not enough points
        return 0.0, {
            'foreground_points_count': len(foreground_points),
            'crop_size': crop.shape[:2],
            'method': 'threshold_failed',
            'threshold_method': threshold_method,
            'min_points_required': min_points
        }
    
    # Center the points
    center = foreground_points.mean(axis=0)
    centered_points = foreground_points - center
    
    # Parameters going into PCA
    pca_params = {
        'foreground_points_count': len(foreground_points),
        'threshold_method': threshold_method,
        'crop_size': crop.shape[:2],
        'bbox_original': {'x1': bbox['x1'], 'y1': bbox['y1'], 'x2': bbox['x2'], 'y2': bbox['y2']},
        'bbox_expanded': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
        'expansion': {'x': expand_x, 'y': expand_y},
        'centered_points_shape': centered_points.shape,
        'center': center.tolist()
    }
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    
    # Get first principal component (direction of maximum variance)
    pc1 = pca.components_[0]
    
    # Calculate angle from horizontal
    # pc1 is in (row, col) format = (y, x) in image coordinates
    angle_rad = np.arctan2(pc1[0], pc1[1])  # atan2(row, col) = atan2(y, x)
    angle_deg = np.degrees(angle_rad)
    
    # Check aspect ratio to determine expected orientation
    crop_h, crop_w = crop.shape[:2]
    aspect_ratio = crop_h / crop_w if crop_w > 0 else 0
    
    # Calculate rotation needed based on orientation
    if aspect_ratio > 1:
        # Vertical strip: PCA angle should be close to ±90°
        rotation_needed = angle_deg - 90.0
    else:
        # Horizontal strip: PCA angle should be close to 0°
        rotation_needed = -angle_deg
    
    # Clamp to reasonable range
    min_angle = pca_cfg.get('min_rotation_angle', -45.0)
    max_angle = pca_cfg.get('max_rotation_angle', 45.0)
    
    if abs(rotation_needed) > 45:
        # If rotation is > 45°, the strip might be in wrong orientation
        # Try the other orientation
        if aspect_ratio > 1:
            rotation_needed = -angle_deg
        else:
            rotation_needed = angle_deg - 90.0
    
    # Clamp final result to reasonable range
    rotation_needed = max(min_angle, min(max_angle, rotation_needed))
    
    # Add PCA results to params
    pca_params.update({
        'pca_components': pca.components_.tolist(),
        'pca_explained_variance': pca.explained_variance_.tolist(),
        'pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'pc1': pca.components_[0].tolist(),
        'angle_rad': float(angle_rad),
        'angle_deg': float(angle_deg),
        'aspect_ratio': float(aspect_ratio),
        'rotation_needed': float(rotation_needed)
    })
    
    return rotation_needed, pca_params




def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate entire image by angle degrees.
    
    Args:
        image: Input image (BGR format)
        angle: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated image
    """
    if abs(angle) < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new dimensions to fit rotated image
    angle_rad = np.radians(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Adjust translation to center the rotated image
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # Rotate image
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White background
    )
    
    return rotated


class StripDetectionService:
    """
    Main service for detecting test strips in images.
    
    Supports multiple detection methods:
    - yolo_pca: YOLO → PCA rotation → Rotate → YOLO (simplified pipeline)
    - yolo_refined: YOLO + refinement pipeline (existing)
    - yolo: YOLO only (no refinement)
    - openai: OpenAI vision API
    - auto: Try methods in order until one succeeds
    """
    
    def __init__(
        self,
        detection_method: Literal['yolo_pca', 'yolo_refined', 'yolo', 'openai', 'auto'] = 'yolo_pca',
        enable_visual_logging: bool = False,
        log_output_dir: Optional[str] = None,
        yolo_detector: Optional[YoloDetector] = None
    ):
        """
        Initialize strip detection service.
        
        Args:
            detection_method: Detection method to use
            enable_visual_logging: Whether to enable visual logging
            log_output_dir: Directory for visual logs (if enabled)
            yolo_detector: Optional pre-initialized YOLO detector
        """
        self.detection_method = detection_method
        self.enable_visual_logging = enable_visual_logging
        self.log_output_dir = log_output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.yolo_detector = yolo_detector or YoloDetector()
        self.strip_refiner = None  # Lazy initialization
        self.openai_service = None  # Lazy initialization
        
        # PCA config
        self.pca_config = get_pca_config()
    
    def detect_strip(
        self,
        image: np.ndarray,
        expected_pad_count: int = 6,
        image_name: str = "unknown",
        use_openai: bool = False,
        debug: Optional[DebugContext] = None
    ) -> Dict:
        """
        Detect test strip in image.
        
        Args:
            image: Input image (BGR format, numpy array)
            expected_pad_count: Expected number of pads (4-7, default: 6)
            image_name: Name of image for logging
            use_openai: Whether to use OpenAI (overrides detection_method if True)
        
        Returns:
            Dictionary with detection results:
            {
                'success': bool,
                'strip_region': {'left': int, 'top': int, 'right': int, 'bottom': int, ...},
                'yolo_bbox': {'x1': int, 'y1': int, 'x2': int, 'y2': int},
                'yolo_confidence': float,
                'method': str,
                'orientation': str,
                'angle': float,
                'visual_log_path': str (optional)
            }
        """
        # Setup debug context
        # Use provided debug context, or create one if visual logging is enabled
        if debug is None and self.enable_visual_logging:
            output_dir = self.log_output_dir or 'experiments/strip_detection'
            debug = DebugContext(
                enabled=True,
                output_dir=output_dir,
                image_name=image_name,
                comparison_tag=self.detection_method
            )
        
        # Determine method
        method = 'openai' if use_openai else self.detection_method
        
        # Log detection method and YOLO availability
        if debug:
            yolo_available = self.yolo_detector.is_available() if self.yolo_detector else False
            debug.add_step(
                '01_00_setup',
                'Detection Setup',
                image,
                {
                    'detection_method': method,
                    'yolo_available': yolo_available,
                    'yolo_model_path': self.yolo_detector.model_path if self.yolo_detector else None,
                    'image_size': {'width': image.shape[1], 'height': image.shape[0]}
                },
                f'Using detection method: {method}, YOLO available: {yolo_available}'
            )
        
        try:
            if method == 'yolo_pca':
                result = self._detect_with_yolo_pca_pipeline(image, debug)
            elif method == 'yolo_refined':
                result = self._detect_with_yolo_refined(image, debug)
            elif method == 'yolo':
                result = self._detect_with_yolo_only(image, debug)
            elif method == 'openai':
                result = self._detect_with_openai(image, debug)
            elif method == 'auto':
                result = self._detect_auto(image, debug)
            else:
                result = {
                    'success': False,
                    'error': f'Unknown detection method: {method}',
                    'error_code': 'UNKNOWN_METHOD'
                }
            
            # Save visual log if enabled
            if debug and debug.is_enabled():
                # Create final visualization from last step or use input image
                final_vis = image.copy()
                if debug.steps:
                    # Use the last step's image as final visualization
                    last_step = debug.steps[-1]
                    if hasattr(last_step, 'image_path') and last_step.image_path:
                        # If step has saved image, we can use it
                        pass
                    # Otherwise, create a simple final visualization
                    # Draw strip region if available
                    if result.get('success') and result.get('strip_region'):
                        strip_region = result['strip_region']
                        cv2.rectangle(
                            final_vis,
                            (strip_region['left'], strip_region['top']),
                            (strip_region['right'], strip_region['bottom']),
                            (0, 255, 0), 3
                        )
                        cv2.putText(
                            final_vis,
                            f"{result.get('method', 'unknown')} - conf: {result.get('yolo_confidence', 0):.2f}",
                            (strip_region['left'], strip_region['top'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                        )
                
                log_path = debug.save_log(final_vis)
                if log_path:
                    result['visual_log_path'] = log_path
            
            return result
            
        except Exception as e:
            self.logger.error(f'Strip detection failed: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'DETECTION_ERROR'
            }
    
    def _detect_with_yolo_pca_pipeline(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext]
    ) -> Dict:
        """
        Detect strip using YOLO → PCA → Rotate → YOLO pipeline.
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Detection result dictionary
        """
        h, w = image.shape[:2]
        
        # Log input image
        if debug:
            debug.add_step(
                '01_01_input',
                'Input Image',
                image,
                {'width': w, 'height': h},
                'Original input image'
            )
        
        # Step 1: YOLO detection on original image
        result1 = self.yolo_detector.detect_strip(image)
        
        if not result1.get('success'):
            if debug:
                debug.add_step(
                    '01_02_yolo_failed',
                    'YOLO Detection Failed',
                    image,
                    {'error': result1.get('error', 'Unknown error')},
                    'YOLO detection failed on original image'
                )
            return {
                'success': False,
                'error': result1.get('error', 'YOLO detection failed'),
                'error_code': 'YOLO_DETECTION_FAILED'
            }
        
        bbox1 = result1['bbox']
        conf1 = result1['confidence']
        
        # Visualize step 1
        if debug:
            vis1 = image.copy()
            cv2.rectangle(vis1, (bbox1['x1'], bbox1['y1']), (bbox1['x2'], bbox1['y2']), (0, 255, 0), 3)
            cv2.putText(vis1, f'YOLO 1 (conf={conf1:.2f})', (bbox1['x1'], bbox1['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug.add_step(
                '01_02_yolo1',
                'YOLO Detection (Original)',
                vis1,
                {
                    'bbox': bbox1,
                    'confidence': conf1,
                    'width': bbox1['x2'] - bbox1['x1'],
                    'height': bbox1['y2'] - bbox1['y1']
                },
                f'YOLO detected strip in original image with confidence {conf1:.3f}, bbox: ({bbox1["x1"]}, {bbox1["y1"]}) to ({bbox1["x2"]}, {bbox1["y2"]})'
            )
        
        # Step 2: PCA rotation detection
        rotation_angle, pca_params = detect_rotation_with_pca(image, bbox1, self.pca_config)
        
        # Visualize PCA detection
        if debug:
            bbox_width = bbox1['x2'] - bbox1['x1']
            bbox_height = bbox1['y2'] - bbox1['y1']
            expand_x = int(bbox_width * self.pca_config.get('horizontal_expansion', 0.10))
            expand_y = int(bbox_height * self.pca_config.get('vertical_expansion', 0.05))
            expanded_x1 = max(0, bbox1['x1'] - expand_x)
            expanded_y1 = max(0, bbox1['y1'] - expand_y)
            expanded_x2 = min(image.shape[1], bbox1['x2'] + expand_x)
            expanded_y2 = min(image.shape[0], bbox1['y2'] + expand_y)
            
            crop_expanded = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
            gray = cv2.cvtColor(crop_expanded, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            vis_pca = image.copy()
            vis_pca[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = binary_colored
            # Draw original YOLO bbox in green
            cv2.rectangle(vis_pca, (bbox1['x1'], bbox1['y1']), (bbox1['x2'], bbox1['y2']), (0, 255, 0), 2)
            # Draw expanded bbox in yellow
            cv2.rectangle(vis_pca, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), (255, 255, 0), 2)
            cv2.putText(vis_pca, f'PCA Angle: {rotation_angle:.2f}°', (expanded_x1, expanded_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            debug.add_step(
                '01_03_pca',
                'PCA Rotation Detection',
                vis_pca,
                {
                    'rotation_angle': rotation_angle,
                    'pca_params': pca_params,
                    'bbox_used': bbox1
                },
                f'PCA analysis detected rotation angle of {rotation_angle:.2f}° from foreground pixel distribution'
            )
        
        # Step 3: Create transform context and apply rotation
        transform_context = ImageTransformContext(image)
        transform_context.apply_rotation(rotation_angle)
        
        if debug:
            current_img = transform_context.get_current_image()
            vis_rotation = current_img.copy()
            h_rot, w_rot = current_img.shape[:2]
            center_x, center_y = w_rot // 2, h_rot // 2
            cv2.line(vis_rotation, (center_x, 0), (center_x, h_rot), (0, 255, 0), 2)
            cv2.putText(vis_rotation, f'Rotated {rotation_angle:.3f}°', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            debug.add_step(
                '01_04_rotated',
                'Rotated Image',
                vis_rotation,
                {
                    'rotation_angle': rotation_angle,
                    'original_size': {'width': w, 'height': h},
                    'rotated_size': {'width': w_rot, 'height': h_rot}
                },
                f'Image rotated by {rotation_angle:.2f}°'
            )
        
        # Step 4: YOLO detection on rotated image (use get_current_image)
        result2 = self.yolo_detector.detect_strip(transform_context.get_current_image())
        
        if not result2.get('success'):
            if debug:
                debug.add_step(
                    '01_05_yolo2_failed',
                    'YOLO Detection (Rotated) - FAILED',
                    transform_context.get_current_image(),
                    {'error': result2.get('error', 'Unknown error')},
                    'YOLO detection failed on rotated image'
                )
            # Fallback: return original detection with rotation info
            strip_region = yolo_result_to_strip_region(result1, image.shape[:2])
            if strip_region:
                return {
                    'success': True,
                    'strip_region': dict(strip_region),
                    'transform_context': transform_context,
                    'yolo_bbox': bbox1,
                    'yolo_confidence': conf1,
                    'method': 'yolo_pca',
                    'orientation': strip_region['orientation'],
                    'angle': rotation_angle,
                    'warning': 'YOLO detection failed on rotated image, using original detection'
                }
            return {
                'success': False,
                'error': 'YOLO detection failed on rotated image',
                'error_code': 'YOLO_ROTATED_FAILED'
            }
        
        bbox2 = result2['bbox']
        conf2 = result2['confidence']
        
        # Step 4.1: Validate YOLO 2 found strip in expected area
        # Transform YOLO 1's bbox to rotated space to get expected location
        expected_bbox_rotated = transform_context.transform_coords_original_to_rotated(bbox1)
        iou = calculate_bbox_iou(bbox2, expected_bbox_rotated)
        
        # Minimum IoU threshold - if YOLO 2 found something in completely wrong place
        min_iou_threshold = self.pca_config.get('yolo2_min_iou', 0.2)
        
        if debug:
            vis2 = transform_context.get_current_image().copy()
            # Draw expected bbox in yellow
            cv2.rectangle(vis2, 
                         (expected_bbox_rotated['x1'], expected_bbox_rotated['y1']), 
                         (expected_bbox_rotated['x2'], expected_bbox_rotated['y2']), 
                         (0, 255, 255), 2)
            cv2.putText(vis2, f'Expected (from YOLO1)', 
                       (expected_bbox_rotated['x1'], expected_bbox_rotated['y1'] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Draw actual detection in blue
            cv2.rectangle(vis2, (bbox2['x1'], bbox2['y1']), (bbox2['x2'], bbox2['y2']), (255, 0, 0), 3)
            cv2.putText(vis2, f'YOLO 2 (conf={conf2:.2f}, IoU={iou:.2f})', (bbox2['x1'], bbox2['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            conf_change = conf2 - conf1
            debug.add_step(
                '01_05_yolo2',
                'YOLO Detection (Rotated)',
                vis2,
                {
                    'bbox': bbox2,
                    'expected_bbox': expected_bbox_rotated,
                    'iou': iou,
                    'confidence': conf2,
                    'width': bbox2['x2'] - bbox2['x1'],
                    'height': bbox2['y2'] - bbox2['y1'],
                    'confidence_change': conf_change
                },
                f'YOLO 2 detected strip with confidence {conf2:.3f}, IoU with expected: {iou:.3f}'
            )
        
        # If IoU is too low, YOLO 2 detected wrong object - crop to expected area and re-run
        if iou < min_iou_threshold:
            self.logger.warning(f'YOLO 2 detection IoU ({iou:.3f}) < threshold ({min_iou_threshold}), re-running on expected area')
            
            # Expand expected bbox by 20% for safety margin
            rotated_h, rotated_w = transform_context.get_current_image().shape[:2]
            expand_margin = 0.3
            exp_w = (expected_bbox_rotated['x2'] - expected_bbox_rotated['x1']) * expand_margin
            exp_h = (expected_bbox_rotated['y2'] - expected_bbox_rotated['y1']) * expand_margin
            
            crop_x1 = max(0, int(expected_bbox_rotated['x1'] - exp_w))
            crop_y1 = max(0, int(expected_bbox_rotated['y1'] - exp_h))
            crop_x2 = min(rotated_w, int(expected_bbox_rotated['x2'] + exp_w))
            crop_y2 = min(rotated_h, int(expected_bbox_rotated['y2'] + exp_h))
            
            # Crop the rotated image to expected area
            cropped_region = transform_context.get_current_image()[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if debug:
                vis_crop = transform_context.get_current_image().copy()
                cv2.rectangle(vis_crop, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 3)
                cv2.putText(vis_crop, 'Cropped region for re-detection', (crop_x1, crop_y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                debug.add_step(
                    '01_05a_yolo2_redetect_region',
                    'YOLO 2 Re-detection Region',
                    vis_crop,
                    {
                        'reason': f'IoU {iou:.3f} < threshold {min_iou_threshold}',
                        'crop_region': {'x1': crop_x1, 'y1': crop_y1, 'x2': crop_x2, 'y2': crop_y2}
                    },
                    f'YOLO 2 found wrong object (IoU={iou:.3f}), cropping to expected area for re-detection'
                )
            
            # Re-run YOLO on cropped region
            result2_crop = self.yolo_detector.detect_strip(cropped_region)
            
            if result2_crop.get('success'):
                # Adjust bbox to full rotated image coordinates
                bbox2_crop = result2_crop['bbox']
                bbox2 = {
                    'x1': bbox2_crop['x1'] + crop_x1,
                    'y1': bbox2_crop['y1'] + crop_y1,
                    'x2': bbox2_crop['x2'] + crop_x1,
                    'y2': bbox2_crop['y2'] + crop_y1
                }
                conf2 = result2_crop['confidence']
                
                if debug:
                    vis2_redetect = transform_context.get_current_image().copy()
                    cv2.rectangle(vis2_redetect, (bbox2['x1'], bbox2['y1']), (bbox2['x2'], bbox2['y2']), (0, 255, 0), 3)
                    cv2.putText(vis2_redetect, f'YOLO 2 Re-detected (conf={conf2:.2f})', (bbox2['x1'], bbox2['y1'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    debug.add_step(
                        '01_05b_yolo2_redetected',
                        'YOLO 2 Re-detected',
                        vis2_redetect,
                        {
                            'bbox': bbox2,
                            'confidence': conf2,
                        },
                        f'YOLO 2 re-detected strip in cropped area with confidence {conf2:.3f}'
                    )
            else:
                # Re-detection failed, use expected bbox directly
                self.logger.warning('YOLO 2 re-detection failed, using expected bbox from YOLO 1')
                bbox2 = expected_bbox_rotated
                conf2 = conf1 * 0.9  # Slightly reduce confidence since we're using transformed bbox
                
                if debug:
                    vis2_fallback = transform_context.get_current_image().copy()
                    cv2.rectangle(vis2_fallback, (bbox2['x1'], bbox2['y1']), (bbox2['x2'], bbox2['y2']), (255, 165, 0), 3)
                    cv2.putText(vis2_fallback, f'Using expected bbox (conf={conf2:.2f})', (bbox2['x1'], bbox2['y1'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    debug.add_step(
                        '01_05b_yolo2_fallback',
                        'YOLO 2 Fallback to Expected',
                        vis2_fallback,
                        {
                            'bbox': bbox2,
                            'confidence': conf2,
                            'reason': 'YOLO 2 re-detection failed, using transformed YOLO 1 bbox'
                        },
                        f'Using expected bbox from YOLO 1 (transformed to rotated space)'
                    )
        
        # Step 4.5: Iterative PCA refinement - run PCA again on rotated image to refine angle
        rotation_angle2 = 0.0
        pca_params2 = {}
        rotation_angle1 = rotation_angle  # Store original rotation for reference
        use_iterative_pca = self.pca_config.get('use_iterative_pca', True)  # Default to True
        
        # Always run second PCA if iterative PCA is enabled (run twice for better accuracy)
        if use_iterative_pca:
            rotation_angle2, pca_params2 = detect_rotation_with_pca(
                transform_context.get_current_image(), bbox2, self.pca_config
            )
            
            # Always apply the refinement (run PCA twice for better accuracy)
            total_rotation = rotation_angle1 + rotation_angle2
            
            if debug:
                vis_pca2 = transform_context.get_current_image().copy()
                cv2.rectangle(vis_pca2, (bbox2['x1'], bbox2['y1']), (bbox2['x2'], bbox2['y2']), (255, 0, 0), 2)
                cv2.putText(vis_pca2, f'PCA2 Angle: {rotation_angle2:.2f}° (Total: {total_rotation:.2f}°)', 
                           (bbox2['x1'], bbox2['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                debug.add_step(
                    '01_06_pca2',
                    'PCA Rotation Refinement',
                    vis_pca2,
                    {
                        'rotation_angle2': rotation_angle2,
                        'rotation_angle1': rotation_angle,
                        'total_rotation': total_rotation,
                        'pca_params2': pca_params2
                    },
                    f'PCA refinement detected additional rotation of {rotation_angle2:.2f}° (total: {total_rotation:.2f}°)'
                )
            
            # Apply refined rotation: replace rotation with total rotation
            # apply_rotation() replaces any existing rotation
            transform_context.apply_rotation(total_rotation)
            rotation_angle = total_rotation
            
            if debug:
                current_img = transform_context.get_current_image()
                vis_rotation2 = current_img.copy()
                h_rot2, w_rot2 = current_img.shape[:2]
                center_x2, center_y2 = w_rot2 // 2, h_rot2 // 2
                cv2.line(vis_rotation2, (center_x2, 0), (center_x2, h_rot2), (0, 255, 0), 2)
                cv2.putText(vis_rotation2, f'Refined Rotation {total_rotation:.3f}°', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                debug.add_step(
                    '01_07_rotated_refined',
                    'Refined Rotated Image',
                    vis_rotation2,
                    {
                        'total_rotation': total_rotation,
                        'rotation_angle1': rotation_angle1,
                        'rotation_angle2': rotation_angle2,
                        'rotated_size': {'width': w_rot2, 'height': h_rot2}
                    },
                    f'Image rotated by refined angle {total_rotation:.2f}°'
                )
            
            # Re-run YOLO on refined rotated image
            result2_refined = self.yolo_detector.detect_strip(transform_context.get_current_image())
            if result2_refined.get('success'):
                bbox2_refined = result2_refined['bbox']
                conf2_refined = result2_refined['confidence']
                
                # Use the refinement
                bbox2 = bbox2_refined
                conf2 = conf2_refined
                
                if debug:
                    vis2_refined = transform_context.get_current_image().copy()
                    cv2.rectangle(vis2_refined, (bbox2['x1'], bbox2['y1']), (bbox2['x2'], bbox2['y2']), (255, 0, 0), 3)
                    cv2.putText(vis2_refined, f'YOLO 2 Refined (conf={conf2:.2f})', (bbox2['x1'], bbox2['y1'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    
                    debug.add_step(
                        '01_08_yolo2_refined',
                        'YOLO Detection (Refined Rotated)',
                        vis2_refined,
                        {
                            'bbox': bbox2,
                            'confidence': conf2,
                            'width': bbox2['x2'] - bbox2['x1'],
                            'height': bbox2['y2'] - bbox2['y1'],
                            'total_rotation': total_rotation,
                            'rotation_angle1': rotation_angle1,
                            'rotation_angle2': rotation_angle2
                        },
                        f'YOLO detected strip in refined rotated image with confidence {conf2:.3f} (total rotation: {total_rotation:.2f}°)'
                    )
            else:
                # YOLO failed on refined image, revert to first rotation
                self.logger.warning('YOLO detection failed on refined rotated image, reverting to first rotation')
                transform_context.apply_rotation(rotation_angle1)
                rotation_angle = rotation_angle1
                
                if debug:
                    debug.add_step(
                        '01_09_pca2_reverted',
                        'PCA Refinement Reverted (YOLO Failed)',
                        transform_context.get_current_image(),
                        {
                            'rotation_angle1': rotation_angle1,
                            'rotation_angle2': rotation_angle2,
                            'total_rotation_attempted': total_rotation,
                            'reason': 'YOLO detection failed on refined image, using first rotation only'
                        },
                        f'YOLO failed on refined image, reverting to first rotation ({rotation_angle1:.2f}°)'
                    )
        
        # Step 5: Refine the bbox on rotated image (preserves rotation from YOLO2)
        # We refine on the rotated image to preserve the rotation information
        refined_strip = None
        if self.strip_refiner is None:
            from config.refinement_config import get_config
            self.strip_refiner = StripRefiner(get_config())
        
        try:
            # Refine on current image (rotated) using bbox2
            refined_strip_rotated = self.strip_refiner.refine(
                transform_context.get_current_image(), bbox2, debug
            )
            
            # Apply crop with padding (this updates transform_context)
            crop_padding = self.pca_config.get('crop_padding', 20)  # Default 20px
            transform_context.apply_crop(refined_strip_rotated, padding=crop_padding)
            
            # Transform refined bbox to original coordinates for the return value
            # Note: After apply_crop, coords are in cropped space, but refined_strip_rotated
            # coords are in rotated space, so we need to transform from rotated to original
            refined_rotated_coords = {
                'left': refined_strip_rotated['left'],
                'top': refined_strip_rotated['top'],
                'right': refined_strip_rotated['right'],
                'bottom': refined_strip_rotated['bottom'],
                'width': refined_strip_rotated['right'] - refined_strip_rotated['left'],
                'height': refined_strip_rotated['bottom'] - refined_strip_rotated['top']
            }
            # Transform from rotated space (before crop) to original
            # We need to temporarily clear crop to transform from rotated space
            saved_crop_offset = transform_context.crop_offset
            saved_cropped_image = transform_context.cropped_image
            transform_context.crop_offset = None
            transform_context.cropped_image = None
            refined_bbox_original = transform_context.transform_coords_to_original(refined_rotated_coords)
            transform_context.crop_offset = saved_crop_offset
            transform_context.cropped_image = saved_cropped_image
            
            # Convert to StripRegion format with original coordinates
            refined_strip = {
                'left': refined_bbox_original['left'],
                'top': refined_bbox_original['top'],
                'right': refined_bbox_original['right'],
                'bottom': refined_bbox_original['bottom'],
                'width': refined_bbox_original['width'],
                'height': refined_bbox_original['height'],
                'confidence': refined_strip_rotated.get('confidence', conf2),
                'detection_method': 'yolo_pca_refined',
                'orientation': refined_strip_rotated.get('orientation', 'vertical'),
                'angle': rotation_angle
            }
            
            # Set metadata
            transform_context.set_metadata('yolo_pca_refined', refined_strip.get('confidence', conf2))
            
            # Visualize refinement result on ROTATED image (before crop)
            if debug:
                vis_refined = transform_context.rotated_image.copy()
                # Draw YOLO2 bbox in blue
                cv2.rectangle(vis_refined, 
                             (bbox2['x1'], bbox2['y1']), 
                             (bbox2['x2'], bbox2['y2']), 
                             (255, 0, 0), 2)
                cv2.putText(vis_refined, f'YOLO2 (conf={conf2:.2f})', 
                           (bbox2['x1'], bbox2['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                # Draw refined bbox in green
                cv2.rectangle(vis_refined,
                             (refined_strip_rotated['left'], refined_strip_rotated['top']),
                             (refined_strip_rotated['right'], refined_strip_rotated['bottom']),
                             (0, 255, 0), 3)
                cv2.putText(vis_refined, f'Refined (rot={rotation_angle:.1f}°)', 
                           (refined_strip_rotated['left'], refined_strip_rotated['top'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                refined_conf = refined_strip_rotated.get('confidence', conf2)
                refined_w = refined_strip_rotated['right'] - refined_strip_rotated['left']
                refined_h = refined_strip_rotated['bottom'] - refined_strip_rotated['top']
                debug.add_step(
                    '01_10_refinement',
                    'Refinement (Rotated Space)',
                    vis_refined,
                    {
                        'yolo2_bbox': bbox2,
                        'yolo2_confidence': conf2,
                        'refined_bbox_rotated': {
                            'left': refined_strip_rotated['left'],
                            'top': refined_strip_rotated['top'],
                            'right': refined_strip_rotated['right'],
                            'bottom': refined_strip_rotated['bottom']
                        },
                        'refined_bbox_original': refined_bbox_original,
                        'refined_confidence': refined_conf,
                        'orientation': refined_strip_rotated.get('orientation', 'vertical'),
                        'rotation_applied': rotation_angle,
                        'crop_offset': transform_context.crop_offset,
                        'crop_shape': transform_context.crop_shape
                    },
                    f'Refined strip region: {refined_w}x{refined_h}px in rotated space, confidence: {refined_conf:.3f}, rotation: {rotation_angle:.2f}°'
                )
        except Exception as e:
            self.logger.warning(f'Refinement failed after YOLO2: {e}', exc_info=True)
            # Fallback: crop using bbox2 directly
            crop_padding = self.pca_config.get('crop_padding', 20)
            transform_context.apply_crop(bbox2, padding=crop_padding)
            
            # Visualize fallback on ROTATED image
            if debug:
                vis_fallback = transform_context.rotated_image.copy()
                cv2.rectangle(vis_fallback, 
                             (bbox2['x1'], bbox2['y1']), 
                             (bbox2['x2'], bbox2['y2']), 
                             (0, 165, 255), 3)
                cv2.putText(vis_fallback, f'Refinement Failed - Using YOLO2 (rot={rotation_angle:.1f}°)', 
                           (bbox2['x1'], bbox2['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                debug.add_step(
                    '01_11_refinement_fallback',
                    'Refinement Failed - Using YOLO2 (Rotated Space)',
                    vis_fallback,
                    {
                        'error': str(e),
                        'yolo2_bbox_rotated': bbox2,
                        'yolo2_confidence': conf2,
                        'rotation_angle': rotation_angle,
                        'warning': 'Refinement failed, using YOLO2-only result'
                    },
                    f'Refinement failed: {str(e)}, using YOLO2 result (rotation preserved: {rotation_angle:.2f}°)'
                )
        
        # Use refined strip if available, otherwise build from YOLO2 result
        # Transform bbox2 from rotated coordinates to original
        # Temporarily clear crop to transform from rotated space
        saved_crop_offset = transform_context.crop_offset
        saved_cropped_image = transform_context.cropped_image
        transform_context.crop_offset = None
        transform_context.cropped_image = None
        
        bbox2_rotated_coords = {
            'left': bbox2['x1'],
            'top': bbox2['y1'],
            'right': bbox2['x2'],
            'bottom': bbox2['y2'],
            'width': bbox2['x2'] - bbox2['x1'],
            'height': bbox2['y2'] - bbox2['y1']
        }
        bbox2_original = transform_context.transform_coords_to_original(bbox2_rotated_coords)
        
        # Restore crop state
        transform_context.crop_offset = saved_crop_offset
        transform_context.cropped_image = saved_cropped_image
        
        if refined_strip:
            strip_region = refined_strip
        else:
            # Fallback to YOLO2 result - use transformed coordinates
            strip_region = {
                'left': bbox2_original['left'],
                'top': bbox2_original['top'],
                'right': bbox2_original['right'],
                'bottom': bbox2_original['bottom'],
                'width': bbox2_original['width'],
                'height': bbox2_original['height'],
                'confidence': conf2,
                'detection_method': 'yolo_pca',
                'orientation': 'vertical' if bbox2_original['height'] > bbox2_original['width'] else 'horizontal',
                'angle': rotation_angle
            }
            transform_context.set_metadata('yolo_pca', conf2)
        
        return {
            'success': True,
            'strip_region': strip_region,
            'transform_context': transform_context,
            'yolo_bbox': bbox2_original,
            'yolo_confidence': conf2,
            'method': 'yolo_pca_refined' if refined_strip else 'yolo_pca',
            'orientation': strip_region['orientation'],
            'angle': strip_region['angle'],
            'rotation_applied': rotation_angle,
            'original_bbox': bbox1,
            'original_confidence': conf1,
            'rotated_bbox': bbox2,
            'refined': refined_strip is not None
        }
    
    def _detect_with_yolo_refined(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext]
    ) -> Dict:
        """
        Detect strip using YOLO + refinement pipeline.
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Detection result dictionary
        """
        # Step 1: YOLO detection
        yolo_result = self.yolo_detector.detect_strip(image)
        
        if not yolo_result.get('success'):
            if debug:
                vis_error = image.copy()
                cv2.putText(vis_error, 'YOLO Detection Failed', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                error_msg = yolo_result.get('error', 'YOLO detection failed')
                cv2.putText(vis_error, error_msg[:50], (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                debug.add_step(
                    '01_02_yolo_failed',
                    'YOLO Detection Failed',
                    vis_error,
                    {
                        'error': error_msg,
                        'error_code': 'YOLO_DETECTION_FAILED',
                        'yolo_available': self.yolo_detector.is_available() if self.yolo_detector else False
                    },
                    'YOLO detection failed on original image'
                )
            return {
                'success': False,
                'error': yolo_result.get('error', 'YOLO detection failed'),
                'error_code': 'YOLO_DETECTION_FAILED'
            }
        
        yolo_bbox = yolo_result['bbox']
        yolo_confidence = yolo_result['confidence']
        
        # Log YOLO detection result
        if debug:
            vis_yolo = image.copy()
            cv2.rectangle(vis_yolo, (yolo_bbox['x1'], yolo_bbox['y1']), 
                         (yolo_bbox['x2'], yolo_bbox['y2']), (0, 255, 0), 3)
            cv2.putText(vis_yolo, f'YOLO (conf={yolo_confidence:.2f})', 
                       (yolo_bbox['x1'], yolo_bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug.add_step(
                '01_02_yolo',
                'YOLO Detection',
                vis_yolo,
                {
                    'bbox': yolo_bbox,
                    'confidence': yolo_confidence,
                    'width': yolo_bbox['x2'] - yolo_bbox['x1'],
                    'height': yolo_bbox['y2'] - yolo_bbox['y1']
                },
                f'YOLO detected strip with confidence {yolo_confidence:.3f}'
            )
        
        # Step 2: Refine with StripRefiner
        if self.strip_refiner is None:
            from config.refinement_config import get_config
            self.strip_refiner = StripRefiner(get_config())
        
        try:
            refined_strip = self.strip_refiner.refine(image, yolo_bbox, debug)
            
            # Convert to dict format
            strip_region = {
                'left': refined_strip['left'],
                'top': refined_strip['top'],
                'right': refined_strip['right'],
                'bottom': refined_strip['bottom'],
                'width': refined_strip['width'],
                'height': refined_strip['height'],
                'confidence': refined_strip['confidence'],
                'detection_method': refined_strip['detection_method'],
                'orientation': refined_strip['orientation'],
                'angle': refined_strip['angle'],
                'prepared_image': refined_strip.get('prepared_image'),  # Include prepared image if available
                'preparation_scale': refined_strip.get('preparation_scale', 1.0)  # Include scale factor
            }
            
            return {
                'success': True,
                'strip_region': strip_region,
                'yolo_bbox': yolo_bbox,
                'yolo_confidence': yolo_confidence,
                'method': 'yolo_refined',
                'orientation': refined_strip['orientation'],
                'angle': refined_strip['angle']
            }
        except Exception as e:
            self.logger.warning(f'Refinement failed: {e}', exc_info=True)
            # Fallback to YOLO-only result
            if debug:
                vis_fallback = image.copy()
                cv2.rectangle(vis_fallback, (yolo_bbox['x1'], yolo_bbox['y1']), 
                             (yolo_bbox['x2'], yolo_bbox['y2']), (0, 165, 255), 3)
                cv2.putText(vis_fallback, 'Refinement Failed - Using YOLO Only', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(vis_fallback, f'YOLO conf={yolo_confidence:.2f}', 
                           (yolo_bbox['x1'], yolo_bbox['y1'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                debug.add_step(
                    '01_10_refinement_fallback',
                    'Refinement Failed - Fallback to YOLO',
                    vis_fallback,
                    {
                        'error': str(e),
                        'yolo_bbox': yolo_bbox,
                        'yolo_confidence': yolo_confidence,
                        'warning': 'Refinement failed, using YOLO-only result'
                    },
                    f'Refinement failed: {str(e)}, falling back to YOLO-only result'
                )
            strip_region = yolo_result_to_strip_region(yolo_result, image.shape[:2])
            if strip_region:
                return {
                    'success': True,
                    'strip_region': dict(strip_region),
                    'yolo_bbox': yolo_bbox,
                    'yolo_confidence': yolo_confidence,
                    'method': 'yolo',
                    'orientation': strip_region['orientation'],
                    'angle': 0.0,
                    'warning': 'Refinement failed, using YOLO-only result'
                }
            return {
                'success': False,
                'error': f'Refinement failed: {str(e)}',
                'error_code': 'REFINEMENT_ERROR'
            }
    
    def _detect_with_yolo_only(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext]
    ) -> Dict:
        """
        Detect strip using YOLO only (no refinement).
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Detection result dictionary
        """
        yolo_result = self.yolo_detector.detect_strip(image)
        
        if not yolo_result.get('success'):
            if debug:
                vis_error = image.copy()
                cv2.putText(vis_error, 'YOLO Detection Failed', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                error_msg = yolo_result.get('error', 'YOLO detection failed')
                cv2.putText(vis_error, error_msg[:50], (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                debug.add_step(
                    '01_02_yolo_failed',
                    'YOLO Detection Failed',
                    vis_error,
                    {
                        'error': error_msg,
                        'error_code': 'YOLO_DETECTION_FAILED',
                        'yolo_available': self.yolo_detector.is_available() if self.yolo_detector else False
                    },
                    'YOLO detection failed'
                )
            return {
                'success': False,
                'error': yolo_result.get('error', 'YOLO detection failed'),
                'error_code': 'YOLO_DETECTION_FAILED'
            }
        
        yolo_bbox = yolo_result['bbox']
        yolo_confidence = yolo_result['confidence']
        
        # Log YOLO detection result
        if debug:
            vis_yolo = image.copy()
            cv2.rectangle(vis_yolo, (yolo_bbox['x1'], yolo_bbox['y1']), 
                         (yolo_bbox['x2'], yolo_bbox['y2']), (0, 255, 0), 3)
            cv2.putText(vis_yolo, f'YOLO (conf={yolo_confidence:.2f})', 
                       (yolo_bbox['x1'], yolo_bbox['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug.add_step(
                '01_02_yolo',
                'YOLO Detection',
                vis_yolo,
                {
                    'bbox': yolo_bbox,
                    'confidence': yolo_confidence,
                    'width': yolo_bbox['x2'] - yolo_bbox['x1'],
                    'height': yolo_bbox['y2'] - yolo_bbox['y1']
                },
                f'YOLO detected strip with confidence {yolo_confidence:.3f}'
            )
        
        strip_region = yolo_result_to_strip_region(yolo_result, image.shape[:2])
        if not strip_region:
            if debug:
                vis_error = image.copy()
                cv2.putText(vis_error, 'Failed to Convert YOLO Result', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                debug.add_step(
                    '01_12_conversion_failed',
                    'Conversion Failed',
                    vis_error,
                    {
                        'error': 'Failed to convert YOLO result',
                        'error_code': 'CONVERSION_ERROR',
                        'yolo_result': yolo_result
                    },
                    'Failed to convert YOLO result to strip region'
                )
            return {
                'success': False,
                'error': 'Failed to convert YOLO result',
                'error_code': 'CONVERSION_ERROR'
            }
        
        return {
            'success': True,
            'strip_region': dict(strip_region),
            'yolo_bbox': yolo_result['bbox'],
            'yolo_confidence': yolo_result['confidence'],
            'method': 'yolo',
            'orientation': strip_region['orientation'],
            'angle': 0.0
        }
    
    def _detect_with_openai(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext]
    ) -> Dict:
        """
        Detect strip using OpenAI vision API.
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Detection result dictionary
        """
        if self.openai_service is None:
            try:
                self.openai_service = OpenAIVisionService()
            except Exception as e:
                self.logger.warning(f'OpenAI service not available: {e}')
                return {
                    'success': False,
                    'error': 'OpenAI service not available',
                    'error_code': 'OPENAI_UNAVAILABLE'
                }
        
        openai_result = self.openai_service.detect_strip(image)
        
        if not openai_result.get('success'):
            return {
                'success': False,
                'error': openai_result.get('error', 'OpenAI detection failed'),
                'error_code': 'OPENAI_DETECTION_FAILED'
            }
        
        strip_region = openai_result_to_strip_region(openai_result)
        if not strip_region:
            return {
                'success': False,
                'error': 'Failed to convert OpenAI result',
                'error_code': 'CONVERSION_ERROR'
            }
        
        return {
            'success': True,
            'strip_region': dict(strip_region),
            'yolo_bbox': None,  # OpenAI doesn't provide YOLO bbox
            'yolo_confidence': openai_result.get('confidence', 0.8),
            'method': 'openai',
            'orientation': strip_region['orientation'],
            'angle': strip_region['angle']
        }
    
    def _detect_auto(
        self,
        image: np.ndarray,
        debug: Optional[DebugContext]
    ) -> Dict:
        """
        Try detection methods in order until one succeeds.
        
        Order: yolo_pca → yolo_refined → yolo → openai
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Detection result dictionary
        """
        methods = ['yolo_pca', 'yolo_refined', 'yolo', 'openai']
        
        for method in methods:
            try:
                if method == 'yolo_pca':
                    result = self._detect_with_yolo_pca_pipeline(image, debug)
                elif method == 'yolo_refined':
                    result = self._detect_with_yolo_refined(image, debug)
                elif method == 'yolo':
                    result = self._detect_with_yolo_only(image, debug)
                elif method == 'openai':
                    result = self._detect_with_openai(image, debug)
                else:
                    continue
                
                if result.get('success'):
                    result['method_used'] = method
                    return result
            except Exception as e:
                self.logger.warning(f'Method {method} failed: {e}')
                continue
        
        return {
            'success': False,
            'error': 'All detection methods failed',
            'error_code': 'ALL_METHODS_FAILED'
        }
