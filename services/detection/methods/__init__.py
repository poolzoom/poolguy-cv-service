"""
Detection methods for strip detection.
"""

from .base_detector import BaseStripDetector
from .canny_detector import CannyDetector
from .color_linear_detector import ColorLinearDetector

# Adaptive threshold detector will be added when extracted
try:
    from .adaptive_threshold_detector import AdaptiveThresholdDetector
except ImportError:
    AdaptiveThresholdDetector = None

__all__ = [
    'BaseStripDetector',
    'CannyDetector',
    'ColorLinearDetector',
]

if AdaptiveThresholdDetector is not None:
    __all__.append('AdaptiveThresholdDetector')



