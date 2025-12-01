"""
Detection Context - Shared state container for strip detection pipeline.

Holds all intermediate results and configuration as detection progresses.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import numpy as np

from services.utils.image_transform_context import ImageTransformContext
from services.utils.debug import DebugContext


@dataclass
class YoloResult:
    """Result from a single YOLO detection attempt."""
    success: bool
    bbox: Optional[Dict] = None
    confidence: float = 0.0
    error: Optional[str] = None
    step_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'error': self.error,
            'step_name': self.step_name
        }


@dataclass
class ValidationResult:
    """Result from validating a detection."""
    is_valid: bool
    reason: str = ""
    iou: float = 0.0
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'reason': self.reason,
            'iou': self.iou,
            'checks_passed': self.checks_passed
        }


@dataclass 
class DetectionContext:
    """
    Holds all state during strip detection pipeline.
    
    This context is passed through each step, accumulating results
    and providing access to shared resources.
    """
    # Input
    original_image: np.ndarray
    config: Dict = field(default_factory=dict)
    debug: Optional[DebugContext] = None
    
    # Transform state
    transform_context: Optional[ImageTransformContext] = None
    
    # Detection results (accumulated)
    yolo1_result: Optional[YoloResult] = None
    yolo2_result: Optional[YoloResult] = None
    yolo2_validation: Optional[ValidationResult] = None
    
    # Rotation state
    rotation_angle: float = 0.0
    rotation_angle_pca1: float = 0.0
    rotation_angle_pca2: float = 0.0
    
    # Expected bbox (transformed from yolo1)
    expected_bbox_rotated: Optional[Dict] = None
    
    # Refinement
    refined_strip: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize transform context if not provided."""
        if self.transform_context is None:
            self.transform_context = ImageTransformContext(self.original_image)
    
    def get_current_image(self) -> np.ndarray:
        """Get the current working image from transform context."""
        return self.transform_context.get_current_image()
    
    def get_image_shape(self) -> tuple:
        """Get shape of original image as (height, width)."""
        return self.original_image.shape[:2]
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a config value with default."""
        return self.config.get(key, default)







