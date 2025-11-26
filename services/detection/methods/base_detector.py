"""
Base class for all strip detection methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np

from utils.visual_logger import VisualLogger

logger = logging.getLogger(__name__)


class BaseStripDetector(ABC):
    """Base class for all strip detection methods."""
    
    def __init__(self, visual_logger: Optional[VisualLogger] = None):
        """
        Initialize detector.
        
        Args:
            visual_logger: Optional visual logger for debugging
        """
        self.visual_logger = visual_logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[Optional[Dict], str, float]:
        """
        Detect test strip in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (strip_region, orientation, angle) or (None, 'vertical', 0.0) if not found
            - strip_region: {'top': int, 'bottom': int, 'left': int, 'right': int}
            - orientation: 'vertical' or 'horizontal'
            - angle: Rotation angle in degrees
        """
        pass
    
    def get_method_name(self) -> str:
        """
        Return method name for logging/identification.
        
        Returns:
            Method name (e.g., 'canny', 'adaptive_threshold', 'color_linear')
        """
        name = self.__class__.__name__.replace('Detector', '').lower()
        # Convert CamelCase to snake_case if needed
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return name



