"""
Base class for refinement methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from services.utils.debug import DebugContext


class BaseRefinementMethod(ABC):
    """Base class for all refinement methods."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize refinement method.
        
        Args:
            config: Configuration dictionary for this method
        """
        self.config = config or {}
    
    @abstractmethod
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Refine the input region.
        
        Args:
            image: Input image (BGR format)
            input_region: Optional input region dict with coordinates
            debug: Optional debug context for logging
            
        Returns:
            Tuple of (refined_region_dict, metadata_dict)
            - refined_region_dict: {'left': int, 'top': int, 'right': int, 'bottom': int}
            - metadata_dict: Additional information about the refinement
        """
        pass
    
    def get_method_name(self) -> str:
        """Return method name for logging."""
        return self.__class__.__name__.replace('Refiner', '').lower()

