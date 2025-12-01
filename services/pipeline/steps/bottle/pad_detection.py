"""
Bottle pad detection service.
Detects colored pads on test strip bottles using YOLO, OpenAI Vision, or OpenCV.

DEPRECATED: This service is no longer used in the new stateless API.
Use TextExtractionService for OpenAI-based detection instead.
Kept for backwards compatibility only.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional

from services.detection.yolo_detector import YoloDetector
from services.detection.openai_vision import OpenAIVisionService
from services.interfaces import PadRegion

logger = logging.getLogger(__name__)


class BottlePadDetectionService:
    """
    Service for detecting pads on test strip bottles.
    
    Uses multiple detection methods with fallback:
    - YOLO (if bottle-trained model available)
    - OpenAI Vision
    - OpenCV contour detection
    """
    
    def __init__(
        self,
        yolo_detector: Optional[YoloDetector] = None,
        openai_service: Optional[OpenAIVisionService] = None,
        detection_method: str = "auto"
    ):
        """
        Initialize bottle pad detection service.
        
        Args:
            yolo_detector: Optional pre-initialized YOLO detector
            openai_service: Optional pre-initialized OpenAI Vision service
            detection_method: Detection method ("yolo", "openai", "opencv", "auto")
        """
        self.logger = logging.getLogger(__name__)
        self.detection_method = detection_method
        
        # Initialize YOLO detector if not provided
        if yolo_detector is None:
            try:
                self.yolo_detector = YoloDetector()
            except Exception as e:
                self.logger.warning(f"YOLO detector not available: {e}")
                self.yolo_detector = None
        else:
            self.yolo_detector = yolo_detector
        
        # Initialize OpenAI service if not provided
        if openai_service is None:
            try:
                self.openai_service = OpenAIVisionService()
            except Exception as e:
                self.logger.warning(f"OpenAI Vision not available: {e}")
                self.openai_service = None
        else:
            self.openai_service = openai_service
    
    def detect_pads(
        self,
        image: np.ndarray,
        debug: Optional[object] = None
    ) -> Dict:
        """
        Detect pads on bottle image.
        
        Args:
            image: Input image (BGR format)
            debug: Optional debug context
        
        Returns:
            Dictionary with detection results:
            {
                'success': bool,
                'pads': [
                    {
                        'pad_index': int,
                        'x': int, 'y': int,
                        'width': int, 'height': int,
                        'left': int, 'top': int,
                        'right': int, 'bottom': int,
                        'confidence': float
                    },
                    ...
                ],
                'detected_count': int,
                'method': str
            }
        """
        # Try methods in order based on detection_method
        methods = []
        if self.detection_method == "auto":
            methods = ["openai", "opencv"]  # Skip YOLO for now (no bottle model)
        elif self.detection_method == "yolo":
            methods = ["yolo"]
        elif self.detection_method == "openai":
            methods = ["openai"]
        elif self.detection_method == "opencv":
            methods = ["opencv"]
        else:
            methods = ["openai", "opencv"]
        
        for method in methods:
            if method == "yolo" and self.yolo_detector:
                result = self._detect_with_yolo(image)
                if result.get('success'):
                    return result
            elif method == "openai" and self.openai_service:
                result = self._detect_with_openai(image)
                if result.get('success'):
                    return result
            elif method == "opencv":
                result = self._detect_with_opencv(image)
                if result.get('success'):
                    return result
        
        # All methods failed
        return {
            'success': False,
            'error': 'Pad detection failed with all methods',
            'error_code': 'PAD_DETECTION_FAILED',
            'pads': [],
            'detected_count': 0,
            'method': 'none'
        }
    
    def _detect_with_yolo(self, image: np.ndarray) -> Dict:
        """
        Detect pads using YOLO (if bottle-trained model available).
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary with detection results
        """
        # Note: This would require a bottle-specific YOLO model
        # For now, return failure
        return {
            'success': False,
            'error': 'YOLO bottle model not available',
            'pads': [],
            'detected_count': 0,
            'method': 'yolo'
        }
    
    def _detect_with_openai(self, image: np.ndarray) -> Dict:
        """
        Detect pads using OpenAI Vision API.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary with detection results
        """
        h, w = image.shape[:2]
        
        prompt = f"""Analyze this test strip bottle image and detect all colored pads.

Image dimensions: {w} x {h} pixels (width x height).

The bottle has colored square/rectangular pads arranged around it. Each pad represents a different chemical test (pH, Chlorine, etc.).

Please identify:
1. The number of pads visible
2. The bounding box for each pad (x, y, width, height in pixels)
3. Order the pads from left to right or top to bottom

Return your response as JSON with this exact format:
{{
    "success": true,
    "pad_count": <integer>,
    "pads": [
        {{
            "pad_index": <integer starting from 0>,
            "x": <integer>,
            "y": <integer>,
            "width": <integer>,
            "height": <integer>,
            "confidence": <float between 0 and 1>
        }},
        ...
    ]
}}

Coordinates are relative to the image (top-left is 0,0).
All coordinates must be within image bounds (0 to {w} for x/width, 0 to {h} for y/height).

If no pads are found, return:
{{
    "success": false,
    "error": "No pads detected"
}}
"""
        
        try:
            result = self.openai_service._make_api_call(image, prompt)
            
            if not result:
                return {
                    'success': False,
                    'error': 'OpenAI API call failed',
                    'pads': [],
                    'detected_count': 0,
                    'method': 'openai'
                }
            
            if not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'No pads detected'),
                    'pads': [],
                    'detected_count': 0,
                    'method': 'openai'
                }
            
            # Validate and convert pads
            pads_data = result.get('pads', [])
            validated_pads = []
            
            for pad in pads_data:
                if not isinstance(pad, dict):
                    continue
                
                # Validate required fields
                if not all(key in pad for key in ['pad_index', 'x', 'y', 'width', 'height']):
                    continue
                
                x = int(pad['x'])
                y = int(pad['y'])
                width = int(pad['width'])
                height = int(pad['height'])
                
                # Validate coordinates
                if not (0 <= x < w and 0 <= y < h and width > 0 and height > 0):
                    continue
                if not (x + width <= w and y + height <= h):
                    continue
                
                pad_region: PadRegion = {
                    'pad_index': int(pad['pad_index']),
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'left': x,
                    'top': y,
                    'right': x + width,
                    'bottom': y + height
                }
                
                validated_pads.append({
                    **pad_region,
                    'confidence': float(pad.get('confidence', 0.8))
                })
            
            # Sort by pad_index
            validated_pads.sort(key=lambda p: p['pad_index'])
            
            return {
                'success': len(validated_pads) > 0,
                'pads': validated_pads,
                'detected_count': len(validated_pads),
                'method': 'openai'
            }
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI pad detection: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'pads': [],
                'detected_count': 0,
                'method': 'openai'
            }
    
    def _detect_with_opencv(self, image: np.ndarray) -> Dict:
        """
        Detect pads using OpenCV contour detection.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            Dictionary with detection results
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for colored regions (exclude white/light backgrounds)
            # Lower saturation threshold to find colored pads
            lower_sat = np.array([50, 50, 50])
            upper_sat = np.array([255, 255, 255])
            mask = cv2.inRange(hsv, lower_sat, upper_sat)
            
            # Apply morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and aspect ratio (pads are roughly square/rectangular)
            min_area = 500  # Minimum pad area
            max_area = image.shape[0] * image.shape[1] * 0.3  # Max 30% of image
            aspect_ratio_range = (0.3, 3.0)  # Allow some variation
            
            detected_pads = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                    continue
                
                pad_region: PadRegion = {
                    'pad_index': i,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'left': x,
                    'top': y,
                    'right': x + w,
                    'bottom': y + h
                }
                
                detected_pads.append({
                    **pad_region,
                    'confidence': 0.7  # Lower confidence for OpenCV method
                })
            
            # Sort by position (left to right, top to bottom)
            detected_pads.sort(key=lambda p: (p['y'], p['x']))
            
            # Reassign pad_index based on sorted order
            for i, pad in enumerate(detected_pads):
                pad['pad_index'] = i
            
            return {
                'success': len(detected_pads) > 0,
                'pads': detected_pads,
                'detected_count': len(detected_pads),
                'method': 'opencv'
            }
            
        except Exception as e:
            self.logger.error(f"Error in OpenCV pad detection: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'pads': [],
                'detected_count': 0,
                'method': 'opencv'
            }



