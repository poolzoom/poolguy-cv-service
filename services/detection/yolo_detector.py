"""
YOLO-based strip detection service for PoolGuy CV Service.
Uses Ultralytics YOLO for object detection to locate test strips in images.
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from config.yolo_config import MODEL_PATH, IMG_SIZE, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class YoloDetector:
    """
    YOLO-based detector for locating test strips in images.
    
    This service uses a pre-trained YOLO model to detect test strip regions
    in images, returning bounding box coordinates and confidence scores.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model file (.pt). If None, uses config default.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If ultralytics is not installed
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics package is not installed. "
                "Install it with: pip install ultralytics"
            )
        
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path or MODEL_PATH
        self.img_size = IMG_SIZE
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Validate model path
        if not os.path.exists(self.model_path):
            self.logger.warning(
                f"YOLO model file not found at {self.model_path}. "
                "YOLO detection will fail until model is trained."
            )
            self.model = None
        else:
            try:
                self.model = YOLO(self.model_path)
                self.logger.info(f"YOLO model loaded from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                self.model = None
    
    def detect_strip(self, image: np.ndarray) -> Dict:
        """
        Detect test strip in image using YOLO.
        
        Args:
            image: Input image as numpy array (BGR format, OpenCV format)
        
        Returns:
            Dictionary with detection results:
            {
                "success": bool,
                "bbox": {
                    "x1": int, "y1": int, "x2": int, "y2": int
                },
                "confidence": float
            }
            If no strip found or model not available, returns success=False.
        """
        if self.model is None:
            return {
                "success": False,
                "bbox": None,
                "confidence": 0.0,
                "error": "YOLO model not loaded"
            }
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                image,
                imgsz=self.img_size,
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Process results
            if len(results) == 0 or len(results[0].boxes) == 0:
                self.logger.debug("YOLO detected no objects")
                return {
                    "success": False,
                    "bbox": None,
                    "confidence": 0.0
                }
            
            # Get the first (highest confidence) detection
            boxes = results[0].boxes
            if len(boxes) == 0:
                return {
                    "success": False,
                    "bbox": None,
                    "confidence": 0.0
                }
            
            # Sort by confidence (descending)
            confidences = boxes.conf.cpu().numpy()
            sorted_indices = np.argsort(confidences)[::-1]
            
            # Get best detection
            best_idx = sorted_indices[0]
            best_box = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]
            best_confidence = float(confidences[best_idx])
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = best_box.astype(int)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            
            self.logger.info(
                f"YOLO detected strip: bbox=({x1}, {y1}, {x2}, {y2}), "
                f"confidence={best_confidence:.3f}"
            )
            
            return {
                "success": True,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": best_confidence
            }
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}", exc_info=True)
            return {
                "success": False,
                "bbox": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """
        Check if YOLO detector is available and ready.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self.model is not None and YOLO_AVAILABLE



