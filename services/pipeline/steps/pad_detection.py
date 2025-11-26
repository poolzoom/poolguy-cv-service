"""
Pad detection service for PoolGuy CV Service.
Detects colored pads within a known strip region using YOLO or multiple strategies.
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from services.interfaces import PadRegion, StripRegion, PadDetectionResult

# Optional YOLO for pad detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Optional OpenAI integration
try:
    from services.detection.openai_vision import OpenAIVisionService
    OPENAI_AVAILABLE = True
except (ImportError, ValueError):
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default pad detection YOLO model path
DEFAULT_PAD_YOLO_MODEL = os.getenv('PAD_YOLO_MODEL_PATH', './models/pad_detection/best.pt')


class PadDetectionService:
    """Service for detecting pad positions within a detected strip region."""
    
    def __init__(self, pad_yolo_model_path: Optional[str] = None, use_yolo: bool = True):
        """
        Initialize pad detection service.
        
        Args:
            pad_yolo_model_path: Path to YOLO pad detection model. If None, uses default.
            use_yolo: Whether to use YOLO for pad detection (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.use_yolo = use_yolo
        self.pad_yolo_model_path = pad_yolo_model_path or DEFAULT_PAD_YOLO_MODEL
        
        # Initialize YOLO model for pad detection
        self.pad_yolo_model = None
        if use_yolo and YOLO_AVAILABLE:
            if os.path.exists(self.pad_yolo_model_path):
                try:
                    self.pad_yolo_model = YOLO(self.pad_yolo_model_path)
                    self.logger.info(f"YOLO pad detection model loaded from {self.pad_yolo_model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load YOLO pad detection model: {e}")
                    self.pad_yolo_model = None
            else:
                self.logger.warning(f"YOLO pad detection model not found at {self.pad_yolo_model_path}, falling back to OpenCV methods")
        
        # Initialize OpenAI service if available
        self.openai_service = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_service = OpenAIVisionService()
                self.logger.info("OpenAI Vision Service available for pad detection")
            except Exception as e:
                self.logger.warning(f"OpenAI Vision Service not available: {e}")
                self.openai_service = None
    
    def _validate_pad_pattern(
        self,
        pads: List[Dict],
        strip_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate pad pattern: check if pads are square, centered, and in repeating pattern.
        
        Args:
            pads: List of pad dictionaries
            strip_image: Strip image for reference
        
        Returns:
            Dictionary with validation results
        """
        if len(pads) == 0:
            return {
                'valid': False,
                'is_square': False,
                'is_centered': False,
                'is_repeating': False,
                'issues': ['No pads to validate']
            }
        
        h, w = strip_image.shape[:2]
        strip_center_x = w / 2
        strip_center_y = h / 2
        
        # Check if pads are square (aspect ratio close to 1.0)
        aspect_ratios = []
        for pad in pads:
            width = pad.get('width', pad.get('right', 0) - pad.get('left', 0))
            height = pad.get('height', pad.get('bottom', 0) - pad.get('top', 0))
            if height > 0:
                aspect_ratio = width / height
                aspect_ratios.append(aspect_ratio)
        
        avg_aspect = np.mean(aspect_ratios) if aspect_ratios else 0
        is_square = 0.7 <= avg_aspect <= 1.3  # Allow some tolerance
        
        # Check if pad centers are near strip center (horizontally)
        center_distances = []
        for pad in pads:
            x = pad.get('x', pad.get('left', 0))
            y = pad.get('y', pad.get('top', 0))
            width = pad.get('width', pad.get('right', 0) - pad.get('left', 0))
            height = pad.get('height', pad.get('bottom', 0) - pad.get('top', 0))
            pad_center_x = x + width / 2
            center_distance = abs(pad_center_x - strip_center_x)
            center_distances.append(center_distance)
        
        max_center_distance = max(center_distances) if center_distances else w
        is_centered = max_center_distance < w * 0.3  # Within 30% of strip width
        
        # Check if pads are in repeating pattern (vertical spacing)
        if len(pads) >= 2:
            # Sort pads by y coordinate
            sorted_pads = sorted(pads, key=lambda p: p.get('y', p.get('top', 0)))
            spacings = []
            for i in range(len(sorted_pads) - 1):
                pad1_bottom = sorted_pads[i].get('y', sorted_pads[i].get('top', 0)) + sorted_pads[i].get('height', 0)
                pad2_top = sorted_pads[i + 1].get('y', sorted_pads[i + 1].get('top', 0))
                spacing = pad2_top - pad1_bottom
                spacings.append(spacing)
            
            if spacings:
                avg_spacing = np.mean(spacings)
                spacing_std = np.std(spacings)
                # Check if spacings are relatively consistent (low std dev)
                is_repeating = spacing_std < avg_spacing * 0.5  # Std dev less than 50% of mean
            else:
                is_repeating = False
        else:
            is_repeating = False
        
        issues = []
        if not is_square:
            issues.append(f'Pads not square (avg aspect ratio: {avg_aspect:.2f})')
        if not is_centered:
            issues.append(f'Pads not centered (max center distance: {max_center_distance:.1f}px)')
        if not is_repeating:
            issues.append('Pads not in repeating pattern')
        
        return {
            'valid': is_square and is_centered and is_repeating,
            'is_square': is_square,
            'is_centered': is_centered,
            'is_repeating': is_repeating,
            'avg_aspect_ratio': avg_aspect,
            'max_center_distance': max_center_distance,
            'issues': issues
        }
    
    def _detect_extra_pads(
        self,
        pads: List[Dict]
    ) -> Dict[str, Any]:
        """
        Detect extra pads: overlapping pads or extra pads between correctly spaced ones.
        
        Args:
            pads: List of pad dictionaries
        
        Returns:
            Dictionary with extra pad detection results
        """
        if len(pads) < 2:
            return {
                'has_overlapping': False,
                'has_extra_between': False,
                'overlapping_pairs': [],
                'extra_pads': []
            }
        
        # Sort pads by y coordinate
        sorted_pads = sorted(pads, key=lambda p: p.get('y', p.get('top', 0)))
        
        # Check for overlapping pads (two on top of each other)
        overlapping_pairs = []
        for i in range(len(sorted_pads)):
            for j in range(i + 1, len(sorted_pads)):
                pad1 = sorted_pads[i]
                pad2 = sorted_pads[j]
                
                x1 = pad1.get('x', pad1.get('left', 0))
                y1 = pad1.get('y', pad1.get('top', 0))
                w1 = pad1.get('width', pad1.get('right', 0) - pad1.get('left', 0))
                h1 = pad1.get('height', pad1.get('bottom', 0) - pad1.get('top', 0))
                
                x2 = pad2.get('x', pad2.get('left', 0))
                y2 = pad2.get('y', pad2.get('top', 0))
                w2 = pad2.get('width', pad2.get('right', 0) - pad2.get('left', 0))
                h2 = pad2.get('height', pad2.get('bottom', 0) - pad2.get('top', 0))
                
                # Calculate intersection
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = x_overlap * y_overlap
                
                # Check if overlap is significant (>30% of smaller pad)
                pad1_area = w1 * h1
                pad2_area = w2 * h2
                min_area = min(pad1_area, pad2_area)
                
                if overlap_area > min_area * 0.3:
                    overlapping_pairs.append((i, j, overlap_area / min_area))
        
        # Check for extra pads between correctly spaced ones
        extra_pads = []
        if len(sorted_pads) >= 3:
            # Calculate expected spacing
            spacings = []
            for i in range(len(sorted_pads) - 1):
                pad1_bottom = sorted_pads[i].get('y', sorted_pads[i].get('top', 0)) + sorted_pads[i].get('height', 0)
                pad2_top = sorted_pads[i + 1].get('y', sorted_pads[i + 1].get('top', 0))
                spacing = pad2_top - pad1_bottom
                spacings.append(spacing)
            
            if spacings:
                avg_spacing = np.mean(spacings)
                # Find pads that are much closer to neighbors than expected
                for i in range(1, len(sorted_pads) - 1):
                    prev_spacing = spacings[i - 1]
                    next_spacing = spacings[i]
                    # If both spacings are much smaller than average, this pad might be extra
                    if prev_spacing < avg_spacing * 0.5 and next_spacing < avg_spacing * 0.5:
                        extra_pads.append(i)
        
        return {
            'has_overlapping': len(overlapping_pairs) > 0,
            'has_extra_between': len(extra_pads) > 0,
            'overlapping_pairs': overlapping_pairs,
            'extra_pads': extra_pads
        }
    
    def _aggregate_detections(
        self,
        strip_image: np.ndarray,
        confidence_thresholds: List[float],
        expected_pad_count: Optional[int] = None
    ) -> List[Dict]:
        """
        Collect all pad detections from multiple confidence thresholds.
        
        Args:
            strip_image: Cropped strip image (BGR format)
            confidence_thresholds: List of confidence thresholds to try
            expected_pad_count: Expected number of pads
        
        Returns:
            List of all detections with their threshold and confidence
        """
        all_detections = []
        
        for conf_thresh in confidence_thresholds:
            yolo_pads = self._detect_pads_with_yolo(strip_image, expected_pad_count, confidence_threshold=conf_thresh)
            
            for pad in yolo_pads:
                # Add metadata about which threshold found this detection
                pad_with_meta = pad.copy()
                pad_with_meta['detection_threshold'] = conf_thresh
                all_detections.append(pad_with_meta)
        
        return all_detections
    
    def _calculate_bbox_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1 = bbox1.get('x', bbox1.get('left', 0))
        y1_1 = bbox1.get('y', bbox1.get('top', 0))
        x2_1 = x1_1 + bbox1.get('width', bbox1.get('right', 0) - x1_1)
        y2_1 = y1_1 + bbox1.get('height', bbox1.get('bottom', 0) - y1_1)
        
        x1_2 = bbox2.get('x', bbox2.get('left', 0))
        y1_2 = bbox2.get('y', bbox2.get('top', 0))
        x2_2 = x1_2 + bbox2.get('width', bbox2.get('right', 0) - x1_2)
        y2_2 = y1_2 + bbox2.get('height', bbox2.get('bottom', 0) - y1_2)
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_center_distance(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate distance between centers of two bounding boxes."""
        x1_1 = bbox1.get('x', bbox1.get('left', 0))
        y1_1 = bbox1.get('y', bbox1.get('top', 0))
        w1 = bbox1.get('width', bbox1.get('right', 0) - x1_1)
        h1 = bbox1.get('height', bbox1.get('bottom', 0) - y1_1)
        center1 = (x1_1 + w1 / 2, y1_1 + h1 / 2)
        
        x1_2 = bbox2.get('x', bbox2.get('left', 0))
        y1_2 = bbox2.get('y', bbox2.get('top', 0))
        w2 = bbox2.get('width', bbox2.get('right', 0) - x1_2)
        h2 = bbox2.get('height', bbox2.get('bottom', 0) - y1_2)
        center2 = (x1_2 + w2 / 2, y1_2 + h2 / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _cluster_detections(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.3,
        center_distance_threshold: float = 50.0
    ) -> List[Dict]:
        """
        Group similar detections across thresholds into clusters.
        
        Args:
            detections: List of all detections from all thresholds
            iou_threshold: Minimum IoU to consider detections similar
            center_distance_threshold: Maximum center distance to consider similar
        
        Returns:
            List of clustered detections (merged from similar detections)
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first) to prioritize better detections
        sorted_detections = sorted(detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
        
        clusters = []
        used = set()
        
        for i, detection in enumerate(sorted_detections):
            if i in used:
                continue
            
            # Start a new cluster with this detection
            cluster = [detection]
            used.add(i)
            
            # Find similar detections
            for j, other in enumerate(sorted_detections[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check if similar (IoU or center distance)
                iou = self._calculate_bbox_iou(detection, other)
                center_dist = self._calculate_center_distance(detection, other)
                
                if iou >= iou_threshold or center_dist <= center_distance_threshold:
                    cluster.append(other)
                    used.add(j)
            
            # Merge cluster into single detection
            if len(cluster) == 1:
                merged = cluster[0].copy()
            else:
                # Weighted average by confidence
                total_conf = sum(d.get('confidence', 0.0) for d in cluster)
                if total_conf > 0:
                    weights = [d.get('confidence', 0.0) / total_conf for d in cluster]
                    x = sum(d.get('x', d.get('left', 0)) * w for d, w in zip(cluster, weights))
                    y = sum(d.get('y', d.get('top', 0)) * w for d, w in zip(cluster, weights))
                    width = sum(d.get('width', 0) * w for d, w in zip(cluster, weights))
                    height = sum(d.get('height', 0) * w for d, w in zip(cluster, weights))
                    conf = max(d.get('confidence', 0.0) for d in cluster)
                else:
                    x = np.mean([d.get('x', d.get('left', 0)) for d in cluster])
                    y = np.mean([d.get('y', d.get('top', 0)) for d in cluster])
                    width = np.mean([d.get('width', 0) for d in cluster])
                    height = np.mean([d.get('height', 0) for d in cluster])
                    conf = 0.5
                
                merged = {
                    'x': int(x),
                    'y': int(y),
                    'width': int(width),
                    'height': int(height),
                    'left': int(x),
                    'top': int(y),
                    'right': int(x + width),
                    'bottom': int(y + height),
                    'confidence': conf,
                    'detection_confidence': conf,
                    'cluster_size': len(cluster),
                    'detection_thresholds': [d.get('detection_threshold', 0.0) for d in cluster]
                }
            
            clusters.append(merged)
        
        # Sort by y-coordinate (top to bottom)
        clusters.sort(key=lambda d: d.get('y', d.get('top', 0)))
        
        return clusters
    
    def _build_pattern_model(
        self,
        clustered_detections: List[Dict],
        strip_image: np.ndarray,
        expected_pad_count: Optional[int] = None
    ) -> Dict:
        """
        Build a pattern model from clustered detections.
        
        Estimates:
        - Pad size (width/height)
        - Strip center line (x-coordinate)
        - Expected spacing between pads
        - Expected pad count
        
        Args:
            clustered_detections: List of clustered detections
            strip_image: Strip image for reference
            expected_pad_count: Expected number of pads
        
        Returns:
            Dictionary with pattern model parameters
        """
        h, w = strip_image.shape[:2]
        strip_center_x = w / 2
        
        if not clustered_detections:
            # Default model if no detections
            estimated_size = min(w, h) * 0.15  # Rough estimate
            return {
                'pad_size': estimated_size,
                'pad_width': estimated_size,
                'pad_height': estimated_size,
                'strip_center_x': strip_center_x,
                'expected_spacing': estimated_size * 1.5,
                'expected_count': expected_pad_count or 5,
                'has_detections': False
            }
        
        # Estimate pad size (median of all detections)
        widths = [d.get('width', 0) for d in clustered_detections if d.get('width', 0) > 0]
        heights = [d.get('height', 0) for d in clustered_detections if d.get('height', 0) > 0]
        
        pad_width = np.median(widths) if widths else w * 0.15
        pad_height = np.median(heights) if heights else h * 0.15
        pad_size = (pad_width + pad_height) / 2
        
        # Estimate strip center (median x-coordinate of pad centers)
        pad_centers_x = []
        for d in clustered_detections:
            x = d.get('x', d.get('left', 0))
            width = d.get('width', 0)
            pad_centers_x.append(x + width / 2)
        
        strip_center_x = np.median(pad_centers_x) if pad_centers_x else w / 2
        
        # Estimate spacing (median gap between consecutive detections)
        sorted_detections = sorted(clustered_detections, key=lambda d: d.get('y', d.get('top', 0)))
        spacings = []
        for i in range(len(sorted_detections) - 1):
            d1 = sorted_detections[i]
            d2 = sorted_detections[i + 1]
            y1_bottom = d1.get('y', d1.get('top', 0)) + d1.get('height', 0)
            y2_top = d2.get('y', d2.get('top', 0))
            gap = y2_top - y1_bottom
            if gap > 0:  # Only positive gaps
                spacings.append(gap)
        
        expected_spacing = np.median(spacings) if spacings else pad_size * 1.5
        
        # Estimate count
        if expected_pad_count:
            estimated_count = expected_pad_count
        else:
            # Infer from pattern: if we have N detections with consistent spacing,
            # estimate total count based on strip height
            if len(clustered_detections) >= 2 and expected_spacing > 0:
                first_y = sorted_detections[0].get('y', sorted_detections[0].get('top', 0))
                last_y = sorted_detections[-1].get('y', sorted_detections[-1].get('top', 0))
                last_bottom = last_y + sorted_detections[-1].get('height', 0)
                total_range = last_bottom - first_y
                estimated_count = int(np.round(total_range / (pad_height + expected_spacing)))
                estimated_count = max(4, min(7, estimated_count))  # Clamp to valid range
            else:
                estimated_count = len(clustered_detections)
        
        return {
            'pad_size': pad_size,
            'pad_width': pad_width,
            'pad_height': pad_height,
            'strip_center_x': strip_center_x,
            'expected_spacing': expected_spacing,
            'expected_count': estimated_count,
            'has_detections': True
        }
    
    def _score_detection(
        self,
        detection: Dict,
        pattern_model: Dict,
        strip_image: np.ndarray,
        position_in_sequence: Optional[int] = None,
        previous_detection: Optional[Dict] = None,
        next_detection: Optional[Dict] = None
    ) -> float:
        """
        Score a detection based on how well it fits the pattern.
        
        Args:
            detection: Detection to score
            pattern_model: Pattern model from _build_pattern_model
            strip_image: Strip image for reference
            position_in_sequence: Position in sequence (0-based)
            previous_detection: Previous detection in sequence (for spacing check)
            next_detection: Next detection in sequence (for spacing check)
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        h, w = strip_image.shape[:2]
        
        x = detection.get('x', detection.get('left', 0))
        y = detection.get('y', detection.get('top', 0))
        width = detection.get('width', 0)
        height = detection.get('height', 0)
        conf = detection.get('confidence', 0.0)
        
        center_x = x + width / 2
        center_y = y + height / 2
        
        scores = []
        
        # 1. Centering score (distance from strip center)
        distance_from_center = abs(center_x - pattern_model['strip_center_x'])
        max_distance = w / 2
        centering_score = max(0.0, 1.0 - (distance_from_center / max_distance))
        scores.append(centering_score * 0.25)  # 25% weight
        
        # 2. Squareness score (aspect ratio close to 1.0)
        if height > 0:
            aspect_ratio = width / height
            squareness_score = 1.0 - min(1.0, abs(1.0 - aspect_ratio))
        else:
            squareness_score = 0.0
        scores.append(squareness_score * 0.20)  # 20% weight
        
        # 3. Size consistency (matches estimated pad size)
        size = (width + height) / 2
        size_ratio = size / pattern_model['pad_size'] if pattern_model['pad_size'] > 0 else 0
        size_score = 1.0 - min(1.0, abs(1.0 - size_ratio))
        scores.append(size_score * 0.20)  # 20% weight
        
        # 4. Spacing consistency (if we have neighbors)
        spacing_score = 1.0
        if previous_detection is not None:
            prev_bottom = previous_detection.get('y', previous_detection.get('top', 0)) + previous_detection.get('height', 0)
            gap = y - prev_bottom
            if pattern_model['expected_spacing'] > 0:
                spacing_ratio = gap / pattern_model['expected_spacing']
                spacing_score = min(spacing_score, 1.0 - min(1.0, abs(1.0 - spacing_ratio)))
        
        if next_detection is not None:
            next_top = next_detection.get('y', next_detection.get('top', 0))
            gap = next_top - (y + height)
            if pattern_model['expected_spacing'] > 0:
                spacing_ratio = gap / pattern_model['expected_spacing']
                spacing_score = min(spacing_score, 1.0 - min(1.0, abs(1.0 - spacing_ratio)))
        
        scores.append(spacing_score * 0.15)  # 15% weight
        
        # 5. Confidence score
        confidence_score = conf
        scores.append(confidence_score * 0.20)  # 20% weight
        
        return sum(scores)
    
    def _calculate_expected_pad_count_from_detections(
        self,
        clustered_detections: List[Dict],
        pattern_model: Dict,
        strip_image: np.ndarray
    ) -> Optional[int]:
        """
        Calculate expected pad count from top-most and bottom-most detected pads.
        
        Uses the fact that:
        1. Pads are square (width ≈ height)
        2. Spacing between pads is approximately equal to pad width
        3. Total distance = (pad_count - 1) * (pad_width + spacing) + pad_width
        
        Args:
            clustered_detections: Detected pads (sorted by y)
            pattern_model: Pattern model with pad_width and expected_spacing
            strip_image: Strip image
        
        Returns:
            Calculated pad count (4-7) or None if calculation is invalid
        """
        if len(clustered_detections) < 2:
            return None  # Need at least 2 pads to calculate
        
        # Sort by y-coordinate
        sorted_detections = sorted(clustered_detections, key=lambda d: d.get('y', d.get('top', 0)))
        
        top_pad = sorted_detections[0]
        bottom_pad = sorted_detections[-1]
        
        # Get pad positions
        top_y = top_pad.get('y', top_pad.get('top', 0))
        bottom_y = bottom_pad.get('y', bottom_pad.get('top', 0))
        bottom_pad_height = bottom_pad.get('height', bottom_pad.get('bottom', 0) - bottom_pad.get('y', bottom_pad.get('top', 0)))
        bottom_pad_bottom = bottom_y + bottom_pad_height
        
        # Calculate total distance from top of first pad to bottom of last pad
        total_distance = bottom_pad_bottom - top_y
        
        # Get pad dimensions and spacing from pattern model
        pad_width = pattern_model.get('pad_width', pattern_model.get('pad_size', 0))
        pad_height = pattern_model.get('pad_height', pattern_model.get('pad_size', 0))
        spacing = pattern_model.get('expected_spacing', 0)
        
        if pad_width <= 0 or pad_height <= 0:
            return None
        
        # User said: spacing ≈ pad_width (approximately equal or a little greater)
        # So we use pad_width as the spacing estimate
        # Each pad takes: pad_height + spacing (where spacing ≈ pad_width)
        # Total: (pad_count - 1) * spacing + pad_count * pad_height
        # total_distance = (pad_count - 1) * spacing + pad_count * pad_height
        # total_distance = pad_count * (spacing + pad_height) - spacing
        # pad_count = (total_distance + spacing) / (spacing + pad_height)
        
        # Use pad_width as spacing (as per user's description)
        effective_spacing = pad_width  # Spacing ≈ pad width
        
        pad_plus_spacing = pad_height + effective_spacing
        if pad_plus_spacing <= 0:
            return None
        
        calculated_count = round((total_distance + effective_spacing) / pad_plus_spacing)
        
        # Debug logging
        self.logger.info(
            f"Pad count calculation: total_distance={total_distance:.1f}px, "
            f"top_pad_y={top_y}, bottom_pad_bottom={bottom_pad_bottom}, "
            f"pad_width={pad_width:.1f}px, pad_height={pad_height:.1f}px, "
            f"effective_spacing={effective_spacing:.1f}px, "
            f"calculated_count={calculated_count}"
        )
        
        # Validate: pad count should be between 4 and 7
        if 4 <= calculated_count <= 7:
            self.logger.info(f"Calculated pad count {calculated_count} is valid")
            return calculated_count
        
        self.logger.warning(f"Calculated pad count {calculated_count} is outside valid range 4-7")
        return None
    
    def _predict_missing_pads(
        self,
        selected_detections: List[Dict],
        pattern_model: Dict,
        strip_image: np.ndarray
    ) -> List[Dict]:
        """
        Predict missing pad locations based on pattern.
        
        Only predicts when gaps are LARGER than expected spacing, indicating a missing pad.
        Does NOT predict when pads are touching or correctly spaced.
        
        Args:
            selected_detections: Currently selected detections (sorted by y)
            pattern_model: Pattern model
            strip_image: Strip image
        
        Returns:
            List of predicted pad locations
        """
        if len(selected_detections) < 2:
            return []  # Need at least 2 detections to infer pattern
        
        h, w = strip_image.shape[:2]
        predicted_pads = []
        
        # Sort by y-coordinate
        sorted_detections = sorted(selected_detections, key=lambda d: d.get('y', d.get('top', 0)))
        
        expected_spacing = pattern_model['expected_spacing']
        pad_height = pattern_model['pad_height']
        # Expected gap between pads = spacing (white space between pads)
        # If a pad is missing, gap = pad_height + spacing (one missing pad + spacing)
        expected_gap_with_missing_pad = pad_height + expected_spacing
        tolerance = expected_spacing * 0.4  # 40% tolerance
        
        # Check for gaps that are LARGER than expected (indicating missing pad)
        for i in range(len(sorted_detections) - 1):
            d1 = sorted_detections[i]
            d2 = sorted_detections[i + 1]
            
            y1_bottom = d1.get('y', d1.get('top', 0)) + d1.get('height', 0)
            y2_top = d2.get('y', d2.get('top', 0))
            gap = y2_top - y1_bottom
            
            # Only predict if gap is significantly larger than expected spacing
            # This means there's likely a missing pad
            # Gap should be approximately: pad_height + spacing (one missing pad + spacing)
            if gap > expected_spacing * 1.5:  # Gap is at least 1.5x spacing (likely has missing pad)
                # Check if gap matches expected gap with one missing pad
                gap_ratio = gap / expected_gap_with_missing_pad
                
                # If gap is approximately 1x (one missing pad) or 2x (two missing pads), predict
                # But be conservative - only predict if gap is clearly larger than normal spacing
                if 0.7 <= gap_ratio <= 2.3:  # Allow for 1 or 2 missing pads
                    # Predict pad(s) in the gap
                    num_missing = round(gap_ratio)
                    if num_missing == 1:
                        # One missing pad - place it in the middle of the gap
                        predicted_y = y1_bottom + gap / 2 - pad_height / 2
                        predicted_x = pattern_model['strip_center_x'] - pattern_model['pad_width'] / 2
                        
                        # Ensure within image bounds
                        predicted_y = max(0, min(h - pad_height, predicted_y))
                        predicted_x = max(0, min(w - pattern_model['pad_width'], predicted_x))
                        
                        predicted_pad = {
                            'x': int(predicted_x),
                            'y': int(predicted_y),
                            'width': int(pattern_model['pad_width']),
                            'height': int(pad_height),
                            'left': int(predicted_x),
                            'top': int(predicted_y),
                            'right': int(predicted_x + pattern_model['pad_width']),
                            'bottom': int(predicted_y + pad_height),
                            'confidence': 0.2,  # Very low confidence for predicted pads
                            'detection_confidence': 0.2,
                            'is_predicted': True
                        }
                        
                        predicted_pads.append(predicted_pad)
                    # For now, only predict one pad at a time (be conservative)
        
        return predicted_pads
    
    def _select_best_combination(
        self,
        clustered_detections: List[Dict],
        pattern_model: Dict,
        strip_image: np.ndarray,
        expected_pad_count: Optional[int] = None,
        debug: Optional[Any] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Select the best combination of detections and predicted pads.
        
        Args:
            clustered_detections: All clustered detections
            pattern_model: Pattern model
            strip_image: Strip image
            expected_pad_count: Expected number of pads
            debug: Debug context
        
        Returns:
            Tuple of (selected_pads, metadata_dict)
        """
        if not clustered_detections:
            return [], {'reason': 'no_detections'}
        
        target_count = expected_pad_count or pattern_model['expected_count']
        
        # Score all detections
        sorted_detections = sorted(clustered_detections, key=lambda d: d.get('y', d.get('top', 0)))
        scored_detections = []
        
        for i, detection in enumerate(sorted_detections):
            prev = sorted_detections[i - 1] if i > 0 else None
            next_d = sorted_detections[i + 1] if i < len(sorted_detections) - 1 else None
            score = self._score_detection(detection, pattern_model, strip_image, i, prev, next_d)
            scored_detections.append((detection, score))
        
        # Sort by score (highest first)
        scored_detections.sort(key=lambda x: x[1], reverse=True)
        
        # Try different combinations
        best_combination = None
        best_score = -1
        best_metadata = {}
        
        # Try different combinations - prefer real detections over predictions
        # Try all possible combinations from 1 to target_count+1 (or available detections)
        max_n = min(len(scored_detections), target_count + 1)
        for n in range(1, max_n + 1):
            selected = [d for d, _ in scored_detections[:n]]
            selected = sorted(selected, key=lambda d: d.get('y', d.get('top', 0)))
            
            # Check if existing detections are well-spaced (no large gaps)
            has_large_gaps = False
            if len(selected) >= 2:
                for i in range(len(selected) - 1):
                    d1 = selected[i]
                    d2 = selected[i + 1]
                    y1_bottom = d1.get('y', d1.get('top', 0)) + d1.get('height', 0)
                    y2_top = d2.get('y', d2.get('top', 0))
                    gap = y2_top - y1_bottom
                    # If gap is much larger than expected spacing, there might be a missing pad
                    if gap > pattern_model['expected_spacing'] * 1.5:
                        has_large_gaps = True
                        break
            
            # Only predict if:
            # 1. We're short of target count
            # 2. We have at least 2 detections (to establish a pattern)
            # 3. There are large gaps indicating missing pads
            predicted = []
            if len(selected) < target_count and len(selected) >= 2 and has_large_gaps:
                predicted = self._predict_missing_pads(selected, pattern_model, strip_image)
                # Only add predictions if we're still short after prediction
                if len(selected) + len(predicted) <= target_count:
                    selected.extend(predicted)
                    selected = sorted(selected, key=lambda d: d.get('y', d.get('top', 0)))
            
            # Score this combination
            combination_score = 0.0
            num_predicted = len([d for d in selected if d.get('is_predicted', False)])
            
            for i, detection in enumerate(selected):
                prev = selected[i - 1] if i > 0 else None
                next_d = selected[i + 1] if i < len(selected) - 1 else None
                detection_score = self._score_detection(detection, pattern_model, strip_image, i, prev, next_d)
                combination_score += detection_score
            
            # Heavy penalty for predicted pads (prefer real detections)
            combination_score -= num_predicted * 0.5
            
            # Bonus for matching expected count
            if len(selected) == target_count:
                combination_score += 2.0
            
            # Penalty for too many or too few
            if len(selected) < target_count - 1 or len(selected) > target_count + 1:
                combination_score -= 1.0
            
            if combination_score > best_score:
                best_score = combination_score
                best_combination = selected
                best_metadata = {
                    'combination_score': combination_score,
                    'detection_count': len([d for d in selected if not d.get('is_predicted', False)]),
                    'predicted_count': len([d for d in selected if d.get('is_predicted', False)]),
                    'total_count': len(selected)
                }
        
        return best_combination or [], best_metadata
    
    def _detect_pads_with_yolo(
        self,
        strip_image: np.ndarray,
        expected_pad_count: Optional[int] = None,
        confidence_threshold: float = 0.20  # Lower threshold to catch more pads
    ) -> List[Dict]:
        """
        Detect pads in cropped strip image using YOLO model.
        
        Args:
            strip_image: Cropped strip image (BGR format)
            expected_pad_count: Expected number of pads (for validation)
            confidence_threshold: Confidence threshold for YOLO detections
        
        Returns:
            List of pad dictionaries with coordinates and confidence
        """
        if self.pad_yolo_model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.pad_yolo_model.predict(
                strip_image,
                conf=confidence_threshold,
                verbose=False
            )
            
            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Get box coordinates (already in pixel coordinates)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Convert to integer coordinates
                    x1_px = int(x1)
                    y1_px = int(y1)
                    x2_px = int(x2)
                    y2_px = int(y2)
                    
                    # Ensure coordinates are within image bounds
                    h, w = strip_image.shape[:2]
                    x1_px = max(0, min(w - 1, x1_px))
                    y1_px = max(0, min(h - 1, y1_px))
                    x2_px = max(x1_px + 1, min(w, x2_px))
                    y2_px = max(y1_px + 1, min(h, y2_px))
                    
                    detections.append({
                        'x': x1_px,
                        'y': y1_px,
                        'width': x2_px - x1_px,
                        'height': y2_px - y1_px,
                        'left': x1_px,
                        'top': y1_px,
                        'right': x2_px,
                        'bottom': y2_px,
                        'confidence': conf,
                        'detection_confidence': conf
                    })
            
            # Sort by y-coordinate (top to bottom)
            detections.sort(key=lambda d: d['y'])
            
            self.logger.info(f"YOLO pad detection found {len(detections)} pads")
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO pad detection failed: {e}", exc_info=True)
            return []
    
    def _filter_white_pads(
        self,
        pads: List[Dict],
        strip_image: np.ndarray,
        expected_pad_count: Optional[int] = None,
        debug: Optional[Any] = None
    ) -> List[Dict]:
        """
        Filter out pads that don't match the expected location pattern.
        
        Test strips have a consistent spacing pattern. If we detect more pads
        than expected, we analyze the spacing pattern to identify and remove
        detections that don't fit (e.g., white spaces between pads).
        
        Args:
            pads: List of pad dictionaries with coordinates (sorted by y)
            strip_image: Cropped strip image (BGR format)
            expected_pad_count: Expected number of pads (4-7)
            
        Returns:
            Filtered list of pads matching the expected pattern
        """
        if not pads or len(pads) <= 1:
            return pads
        
        # If we have the expected count or fewer, no filtering needed
        if expected_pad_count and len(pads) <= expected_pad_count:
            return pads
        
        # If we have more pads than expected, analyze spacing pattern
        # Pads should have consistent spacing with positive gaps between them
        # Calculate gaps between consecutive pads
        gaps = []
        overlaps = []  # Track which pads overlap with neighbors
        for i in range(len(pads) - 1):
            pad1 = pads[i]
            pad2 = pads[i + 1]
            # Get pad boundaries
            pad1_top = pad1.get('y', pad1.get('top', 0))
            pad1_bottom = pad1_top + pad1.get('height', pad1.get('bottom', 0) - pad1_top)
            pad2_top = pad2.get('y', pad2.get('top', 0))
            pad2_bottom = pad2_top + pad2.get('height', pad2.get('bottom', 0) - pad2_top)
            
            # Gap is distance between bottom of pad1 and top of pad2
            gap = pad2_top - pad1_bottom
            gaps.append(gap)
            
            # Check for overlap (negative gap means overlap)
            if gap < 0:
                # Pad2 overlaps with pad1 - mark both as potentially problematic
                overlaps.append(i + 1)  # pad2 index
                if i not in overlaps:
                    overlaps.append(i)  # pad1 index
        
        if len(gaps) == 0:
            return pads
        
        # Filter out pads that overlap with neighbors (likely false detections)
        # A pad that overlaps with both neighbors is definitely a false detection
        filtered_pads = []
        for i, pad in enumerate(pads):
            # Check if this pad overlaps with neighbors
            overlaps_before = False
            overlaps_after = False
            
            if i > 0:
                pad_before = pads[i - 1]
                pad_before_bottom = pad_before.get('y', pad_before.get('top', 0)) + pad_before.get('height', pad_before.get('bottom', 0) - pad_before.get('y', pad_before.get('top', 0)))
                pad_top = pad.get('y', pad.get('top', 0))
                overlaps_before = pad_top < pad_before_bottom
            
            if i < len(pads) - 1:
                pad_after = pads[i + 1]
                pad_bottom = pad.get('y', pad.get('top', 0)) + pad.get('height', pad.get('bottom', 0) - pad.get('y', pad.get('top', 0)))
                pad_after_top = pad_after.get('y', pad_after.get('top', 0))
                overlaps_after = pad_bottom > pad_after_top
            
            # If pad overlaps with both neighbors, it's likely white space (false detection)
            if overlaps_before and overlaps_after:
                self.logger.info(f"Filtering out pad at index {i} (y={pad.get('y', 'N/A')}) - overlaps with both neighbors (white space)")
                continue
            
            filtered_pads.append(pad)
        
        # If we still have more than expected after overlap filtering, use gap analysis
        if expected_pad_count and len(filtered_pads) > expected_pad_count:
            # Calculate average gap from remaining pads
            remaining_gaps = []
            for i in range(len(filtered_pads) - 1):
                pad1 = filtered_pads[i]
                pad2 = filtered_pads[i + 1]
                pad1_bottom = pad1.get('y', pad1.get('top', 0)) + pad1.get('height', pad1.get('bottom', 0) - pad1.get('y', pad1.get('top', 0)))
                pad2_top = pad2.get('y', pad2.get('top', 0))
                gap = pad2_top - pad1_bottom
                if gap > 0:  # Only count positive gaps
                    remaining_gaps.append(gap)
            
            if len(remaining_gaps) > 0:
                avg_gap = np.mean(remaining_gaps)
                # Find pad with largest gap before it (likely white space before it)
                gaps_before_pads = []
                for i in range(len(filtered_pads)):
                    if i == 0:
                        gaps_before_pads.append(0)
                    else:
                        pad1 = filtered_pads[i - 1]
                        pad2 = filtered_pads[i]
                        pad1_bottom = pad1.get('y', pad1.get('top', 0)) + pad1.get('height', pad1.get('bottom', 0) - pad1.get('y', pad1.get('top', 0)))
                        pad2_top = pad2.get('y', pad2.get('top', 0))
                        gap = pad2_top - pad1_bottom
                        gaps_before_pads.append(gap)
                
                # Find pad with largest gap before it (if > 2x average)
                max_gap_before_idx = np.argmax(gaps_before_pads[1:]) + 1  # Skip first pad (no gap before it)
                max_gap_before = gaps_before_pads[max_gap_before_idx]
                
                if max_gap_before > avg_gap * 2.5:  # More aggressive threshold
                    removed_pad = filtered_pads[max_gap_before_idx]
                    self.logger.info(f"Filtering out pad at index {max_gap_before_idx} (y={removed_pad.get('y', 'N/A')}) - large gap before indicates white space")
                    filtered_pads = [p for i, p in enumerate(filtered_pads) if i != max_gap_before_idx]
        
        return filtered_pads if len(filtered_pads) > 0 else pads
        
        # If no clear pattern violation, return original pads
        return pads
    
    def detect_pads_in_strip(
        self,
        strip_image: np.ndarray,
        strip_region: StripRegion,
        expected_pad_count: Optional[int] = None,
        debug: Optional[Any] = None
    ) -> PadDetectionResult:
        """
        Detect pads within a pre-cropped strip image.
        
        This is the new interface that accepts a cropped strip image directly.
        All coordinates in the returned PadRegion objects are relative to the strip_image,
        and must be transformed to absolute coordinates by the caller.
        
        Args:
            strip_image: Pre-cropped strip image (BGR format)
            strip_region: StripRegion object with absolute coordinates in original image
            expected_pad_count: Expected number of pads (4-7)
            
        Returns:
            PadDetectionResult with list of PadRegion objects (relative to strip_image)
        """
        if strip_image.size == 0:
            return PadDetectionResult(
                success=False,
                pads=[],
                error="Strip image is empty",
                error_code="EMPTY_STRIP_IMAGE",
                detected_count=0
            )
        
        try:
            h, w = strip_image.shape[:2]
            
            # Debug: Log input image info
            if debug:
                debug.add_step(
                    '02_00_input',
                    'Pad Detection Input',
                    strip_image,
                    {
                        'image_shape': {'width': w, 'height': h},
                        'expected_pad_count': expected_pad_count,
                        'yolo_model_available': self.pad_yolo_model is not None,
                        'yolo_model_path': self.pad_yolo_model_path if self.pad_yolo_model else None
                    },
                    f'Input strip image: {w}x{h}px, expected {expected_pad_count} pads'
                )
            
            # Try YOLO pad detection first if available
            if self.pad_yolo_model is not None:
                # NEW PATTERN-BASED APPROACH
                # Step 1: Aggregate all detections from all thresholds
                confidence_thresholds = [0.15, 0.20, 0.25, 0.30]
                all_detections = self._aggregate_detections(strip_image, confidence_thresholds, expected_pad_count)
                
                # Debug: Show raw detections from each threshold
                if debug:
                    for conf_thresh in confidence_thresholds:
                        threshold_detections = [d for d in all_detections if d.get('detection_threshold') == conf_thresh]
                        vis_thresh = strip_image.copy()
                        for i, pad in enumerate(threshold_detections):
                            x = pad.get('x', pad.get('left', 0))
                            y = pad.get('y', pad.get('top', 0))
                            width = pad.get('width', pad.get('right', 0) - x)
                            height = pad.get('height', pad.get('bottom', 0) - y)
                            conf = pad.get('confidence', 0.0)
                            cv2.rectangle(vis_thresh, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            cv2.putText(vis_thresh, f'P{i+1} {conf:.2f}', (x, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        cv2.putText(vis_thresh, f'Conf Threshold: {conf_thresh}', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(vis_thresh, f'Detected: {len(threshold_detections)} pads', (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        debug.add_step(
                            f'02_01_yolo_conf_{conf_thresh}',
                            f'YOLO Conf={conf_thresh}',
                            vis_thresh,
                            {
                                'threshold': conf_thresh,
                                'detected_count': len(threshold_detections)
                            },
                            f'Conf={conf_thresh}: {len(threshold_detections)} pads detected'
                        )
                
                # Step 2: Cluster similar detections
                clustered_detections = self._cluster_detections(all_detections)
                
                # Debug: Show clustered detections
                if debug:
                    vis_clustered = strip_image.copy()
                    for i, pad in enumerate(clustered_detections):
                        x = pad.get('x', pad.get('left', 0))
                        y = pad.get('y', pad.get('top', 0))
                        width = pad.get('width', pad.get('right', 0) - x)
                        height = pad.get('height', pad.get('bottom', 0) - y)
                        conf = pad.get('confidence', 0.0)
                        cluster_size = pad.get('cluster_size', 1)
                        color = (0, 255, 255) if cluster_size > 1 else (0, 255, 0)
                        cv2.rectangle(vis_clustered, (x, y), (x + width, y + height), color, 2)
                        label = f'C{i+1} {conf:.2f}'
                        if cluster_size > 1:
                            label += f' ({cluster_size})'
                        cv2.putText(vis_clustered, label, (x, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(vis_clustered, f'Clustered: {len(clustered_detections)} unique pads', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    debug.add_step(
                        '02_02_clustered',
                        'Clustered Detections',
                        vis_clustered,
                        {
                            'cluster_count': len(clustered_detections),
                            'original_detection_count': len(all_detections)
                        },
                        f'Clustered {len(all_detections)} detections into {len(clustered_detections)} unique pads'
                    )
                
                # Step 3: Build pattern model
                pattern_model = self._build_pattern_model(clustered_detections, strip_image, expected_pad_count)
                
                # Step 3.5: Calculate expected pad count from detected pads
                # If we have top and bottom pads, we can estimate total count
                calculated_pad_count = self._calculate_expected_pad_count_from_detections(
                    clustered_detections, pattern_model, strip_image
                )
                
                # Use calculated count if it differs from expected and is valid
                effective_pad_count = expected_pad_count
                if calculated_pad_count is not None and calculated_pad_count != expected_pad_count:
                    self.logger.info(
                        f"Overriding expected pad count: {expected_pad_count} -> {calculated_pad_count} "
                        f"(calculated from detected pad positions)"
                    )
                    effective_pad_count = calculated_pad_count
                    # Update pattern model with new expected count
                    pattern_model['expected_count'] = calculated_pad_count
                
                # Debug: Show pattern model
                if debug:
                    vis_pattern = strip_image.copy()
                    h, w = strip_image.shape[:2]
                    
                    # Draw strip center line
                    center_x = int(pattern_model['strip_center_x'])
                    cv2.line(vis_pattern, (center_x, 0), (center_x, h), (255, 0, 0), 2)
                    
                    # Draw estimated pad size
                    pad_size = pattern_model['pad_size']
                    cv2.putText(vis_pattern, f'Pattern Model:', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(vis_pattern, f'Pad Size: {pad_size:.1f}px', (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(vis_pattern, f'Spacing: {pattern_model["expected_spacing"]:.1f}px', (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(vis_pattern, f'Expected Count: {pattern_model["expected_count"]}', (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show override info if calculated count differs
                    if calculated_pad_count is not None and calculated_pad_count != expected_pad_count:
                        cv2.putText(vis_pattern, f'Override: {expected_pad_count} -> {calculated_pad_count}', (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    pattern_data = pattern_model.copy()
                    pattern_data['original_expected_count'] = expected_pad_count
                    pattern_data['calculated_pad_count'] = calculated_pad_count
                    pattern_data['effective_pad_count'] = effective_pad_count
                    
                    debug.add_step(
                        '02_03_pattern_model',
                        'Pattern Model',
                        vis_pattern,
                        pattern_data,
                        f'Pattern: {effective_pad_count} pads, size={pattern_model["pad_size"]:.1f}px, spacing={pattern_model["expected_spacing"]:.1f}px'
                    )
                
                # Step 4: Select best combination
                selected_pads, selection_metadata = self._select_best_combination(
                    clustered_detections, pattern_model, strip_image, effective_pad_count, debug
                )
                
                # Debug: Show final selection
                if debug:
                    vis_final = strip_image.copy()
                    for i, pad in enumerate(selected_pads):
                        x = pad.get('x', pad.get('left', 0))
                        y = pad.get('y', pad.get('top', 0))
                        width = pad.get('width', pad.get('right', 0) - x)
                        height = pad.get('height', pad.get('bottom', 0) - y)
                        is_predicted = pad.get('is_predicted', False)
                        color = (0, 165, 255) if is_predicted else (0, 255, 0)  # Orange for predicted, green for detected
                        cv2.rectangle(vis_final, (x, y), (x + width, y + height), color, 2)
                        label = f'P{i+1}'
                        if is_predicted:
                            label += ' (pred)'
                        cv2.putText(vis_final, label, (x, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.putText(vis_final, f'Selected: {len(selected_pads)} pads', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(vis_final, f'Detected: {selection_metadata.get("detection_count", 0)}, Predicted: {selection_metadata.get("predicted_count", 0)}', (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    debug.add_step(
                        '02_04_final_selection',
                        'Final Selection (Pattern-Based)',
                        vis_final,
                        selection_metadata,
                        f'Selected {len(selected_pads)} pads (detected: {selection_metadata.get("detection_count", 0)}, predicted: {selection_metadata.get("predicted_count", 0)})'
                    )
                
                # Convert to PadRegion format
                if selected_pads:
                    pad_regions: List[PadRegion] = []
                    for idx, pad_dict in enumerate(selected_pads):
                        x = pad_dict.get('x', pad_dict.get('left', 0))
                        y = pad_dict.get('y', pad_dict.get('top', 0))
                        width = pad_dict.get('width', pad_dict.get('right', 0) - x)
                        height = pad_dict.get('height', pad_dict.get('bottom', 0) - y)
                        
                        pad_regions.append(PadRegion(
                            pad_index=idx,
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            left=x,
                            top=y,
                            right=x + width,
                            bottom=y + height
                        ))
                    
                    # Accept if in valid range (4-7) or if we have at least 1 valid detection
                    # (Better to return partial results than fail completely)
                    if 4 <= len(pad_regions) <= 7:
                        if effective_pad_count and len(pad_regions) != effective_pad_count:
                            self.logger.info(f"Pattern-based detection found {len(pad_regions)} pads (expected {effective_pad_count})")
                        return PadDetectionResult(
                            success=True,
                            pads=pad_regions,
                            error=None,
                            error_code=None,
                            detected_count=len(pad_regions)
                        )
                    elif len(pad_regions) > 0:
                        # Accept partial results (1-3 pads) but log warning
                        self.logger.warning(f"Pattern-based detection found only {len(pad_regions)} pad(s) (expected {effective_pad_count or 'unknown'}, valid range 4-7)")
                        return PadDetectionResult(
                            success=True,  # Still return success so pipeline can continue
                            pads=pad_regions,
                            error=None,
                            error_code=None,
                            detected_count=len(pad_regions)
                        )
                    else:
                        # No pads at all
                        self.logger.warning(f"Pattern-based detection found {len(pad_regions)} pads (outside valid range 4-7)")
                        return PadDetectionResult(
                            success=False,
                            pads=pad_regions,
                            error=f'Pattern-based detection found {len(pad_regions)} pads (outside valid range 4-7)',
                            error_code='INVALID_COUNT',
                            detected_count=len(pad_regions)
                        )
                else:
                    # No pads selected
                    if debug:
                        debug.add_step(
                            '02_05_no_selection',
                            'No Pads Selected',
                            strip_image,
                            {'reason': 'No valid pad combination found'},
                            'Pattern-based selection found no valid pad combination'
                        )
                    return PadDetectionResult(
                        success=False,
                        pads=[],
                        error='Pattern-based detection found no valid pads',
                        error_code='NO_PADS',
                        detected_count=0
                    )
            else:
                # YOLO model not available - fail cleanly
                if debug:
                    debug.add_step(
                        '02_08_yolo_unavailable',
                        'YOLO Model Unavailable',
                        strip_image,
                        {
                            'yolo_model_available': False,
                            'error': 'YOLO pad detection model not loaded'
                        },
                        'YOLO pad detection model not available'
                    )
                return PadDetectionResult(
                    success=False,
                    pads=[],
                    error='YOLO pad detection model not available',
                    error_code='YOLO_MODEL_UNAVAILABLE',
                    detected_count=0
                )
            
        except Exception as e:
            self.logger.error(f"Pad detection failed: {e}", exc_info=True)
            return PadDetectionResult(
                success=False,
                pads=[],
                error=str(e),
                error_code="PAD_DETECTION_ERROR",
                detected_count=0
            )
