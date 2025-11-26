"""
Detection services and utilities.

Detectors:
- YoloDetector: YOLO-based strip detection
- OpenAIVisionService: OpenAI vision API for detection

Utilities:
- detector_adapters: Convert detector outputs to standardized format
- methods: Base detection methods (Canny, color linear, etc.)
"""

from services.detection.yolo_detector import YoloDetector
from services.detection.openai_vision import OpenAIVisionService

__all__ = [
    'YoloDetector',
    'OpenAIVisionService'
]


