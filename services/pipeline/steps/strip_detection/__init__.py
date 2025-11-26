"""
Strip Detection Package - Multi-step pipeline for detecting test strips.

This package provides a clean, modular strip detection pipeline:
- service.py: Main orchestrator (StripDetectionService)
- context.py: Shared state container (DetectionContext)
- validation.py: Detection validation logic
- yolo_steps.py: YOLO detection step helpers
- pca_steps.py: PCA rotation step helpers

Usage:
    from services.pipeline.steps.strip_detection import StripDetectionService
    
    service = StripDetectionService(detection_method='yolo_pca')
    result = service.detect_strip(image)
"""

from .service import StripDetectionService
from .context import DetectionContext, YoloResult, ValidationResult
from .validation import validate_yolo_result, calculate_bbox_iou

__all__ = [
    'StripDetectionService',
    'DetectionContext',
    'YoloResult',
    'ValidationResult',
    'validate_yolo_result',
    'calculate_bbox_iou'
]

