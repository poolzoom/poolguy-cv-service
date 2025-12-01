"""
Bottle pipeline steps for test strip bottle processing.

Active Services:
- TextExtractionService: OpenAI-based text/structure extraction
- SwatchDetectionService: OpenCV-based swatch detection (no AI)

Deprecated Services (kept for backwards compatibility):
- BottlePadDetectionService: Use TextExtractionService instead
- ReferenceSquareDetectionService: Alias for SwatchDetectionService
- ColorMappingService: Not needed in new stateless API
"""

from .text_extraction import TextExtractionService
from .reference_square_detection import SwatchDetectionService

# Legacy aliases for backwards compatibility
from .reference_square_detection import ReferenceSquareDetectionService
from .pad_detection import BottlePadDetectionService
from .color_mapping import ColorMappingService

__all__ = [
    # Active services
    'TextExtractionService',
    'SwatchDetectionService',
    # Legacy aliases
    'ReferenceSquareDetectionService',
    'BottlePadDetectionService',
    'ColorMappingService'
]
