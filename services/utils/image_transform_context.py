"""
Image Transform Context - Simplified transform chain for pipeline processing.

Supports two transform types:
1. ROTATION - Rotate the original image by an angle
2. SELECTION/CROP - Select a region from the rotated image

Each step calls get_current_image() to work on the latest output.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ImageTransformContext:
    """
    Manages image transformations through the pipeline.
    
    Transform chain:
        Original Image (immutable)
               │
           [ROTATION] ← angle, matrices
               │
           Rotated Image (cached)
               │
           [SELECTION] ← offset (x, y)
               │
           Cropped Image ← get_current_image()
    """
    
    def __init__(self, original_image: np.ndarray):
        """
        Initialize transform context with original image.
        
        Args:
            original_image: Original input image (BGR format)
        """
        # Immutable reference
        self.original_image = original_image
        self.original_shape = original_image.shape[:2]  # (height, width)
        
        # Rotation state
        self.rotated_image: Optional[np.ndarray] = None
        self.rotation_angle: float = 0.0
        self.rotation_applied: bool = False
        self.rotation_matrix: Optional[np.ndarray] = None
        self.inverse_rotation_matrix: Optional[np.ndarray] = None
        self.rotated_shape: Optional[Tuple[int, int]] = None
        
        # Crop state
        self.cropped_image: Optional[np.ndarray] = None
        self.crop_offset: Optional[Tuple[int, int]] = None  # (x, y) in rotated space
        self.crop_shape: Optional[Tuple[int, int]] = None
        
        # Metadata
        self.detection_method: str = "unknown"
        self.confidence: float = 0.0
    
    def apply_rotation(self, angle: float) -> None:
        """
        Rotate the original image by the specified angle.
        
        Replaces any existing rotation. Clears any existing crop.
        
        Args:
            angle: Rotation angle in degrees (positive = counterclockwise)
        """
        # Clear any existing crop (rotation replaces entire transform state)
        self.cropped_image = None
        self.crop_offset = None
        self.crop_shape = None
        
        if abs(angle) < 0.01:
            # No rotation needed
            self.rotated_image = self.original_image
            self.rotation_angle = 0.0
            self.rotation_applied = False
            self.rotation_matrix = None
            self.inverse_rotation_matrix = None
            self.rotated_shape = self.original_shape
            return
        
        h, w = self.original_shape
        center = (w / 2, h / 2)
        
        # Calculate new dimensions to fit rotated image
        angle_rad = np.radians(angle)
        cos_a = abs(np.cos(angle_rad))
        sin_a = abs(np.sin(angle_rad))
        
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Adjust translation to center the rotated image
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Rotate image
        rotated = cv2.warpAffine(
            self.original_image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # White background
        )
        
        # Compute inverse rotation matrix for coordinate transforms
        # Inverse rotates from rotated space back to original space
        inv_center = (new_w / 2, new_h / 2)
        inverse_matrix = cv2.getRotationMatrix2D(inv_center, -angle, 1.0)
        # Adjust translation for original dimensions
        inverse_matrix[0, 2] += (w - new_w) / 2
        inverse_matrix[1, 2] += (h - new_h) / 2
        
        # Store state
        self.rotated_image = rotated
        self.rotation_angle = angle
        self.rotation_applied = True
        self.rotation_matrix = rotation_matrix
        self.inverse_rotation_matrix = inverse_matrix
        self.rotated_shape = (new_h, new_w)
        
        logger.debug(f'Applied rotation: {angle:.2f}°, original: {w}x{h}, rotated: {new_w}x{new_h}')
    
    def apply_crop(self, bbox: Dict, padding: int = 0) -> None:
        """
        Crop a region from the current image (rotated or original).
        
        Args:
            bbox: Bounding box dict with 'x1', 'y1', 'x2', 'y2' or 'left', 'top', 'right', 'bottom'
            padding: Extra padding around the bbox (default: 0)
        """
        # Get the image to crop from
        source_image = self.rotated_image if self.rotated_image is not None else self.original_image
        source_h, source_w = source_image.shape[:2]
        
        # Normalize bbox format
        x1 = bbox.get('x1', bbox.get('left', 0))
        y1 = bbox.get('y1', bbox.get('top', 0))
        x2 = bbox.get('x2', bbox.get('right', source_w))
        y2 = bbox.get('y2', bbox.get('bottom', source_h))
        
        # Apply padding and clamp to image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(source_w, x2 + padding)
        y2 = min(source_h, y2 + padding)
        
        # Crop the image
        self.cropped_image = source_image[y1:y2, x1:x2].copy()
        self.crop_offset = (x1, y1)
        self.crop_shape = (y2 - y1, x2 - x1)
        
        logger.debug(f'Applied crop: offset=({x1}, {y1}), size={x2-x1}x{y2-y1}, padding={padding}')
    
    def expand_crop(self, padding: int) -> None:
        """
        Expand the current crop by going back to the rotated image and re-cropping.
        
        Args:
            padding: Additional padding to add around current crop bounds
        """
        if self.crop_offset is None:
            logger.warning('No crop to expand')
            return
        
        # Get the image to crop from
        source_image = self.rotated_image if self.rotated_image is not None else self.original_image
        source_h, source_w = source_image.shape[:2]
        
        # Calculate expanded bounds
        x1, y1 = self.crop_offset
        x2 = x1 + self.crop_shape[1]  # width
        y2 = y1 + self.crop_shape[0]  # height
        
        # Expand and clamp
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(source_w, x2 + padding)
        y2 = min(source_h, y2 + padding)
        
        # Re-crop
        self.cropped_image = source_image[y1:y2, x1:x2].copy()
        self.crop_offset = (x1, y1)
        self.crop_shape = (y2 - y1, x2 - x1)
        
        logger.debug(f'Expanded crop: offset=({x1}, {y1}), size={x2-x1}x{y2-y1}')
    
    def get_current_image(self) -> np.ndarray:
        """
        Get the current working image (result of all transforms).
        
        Returns:
            - cropped_image if crop has been applied
            - rotated_image if only rotation has been applied
            - original_image if no transforms applied
        """
        if self.cropped_image is not None:
            return self.cropped_image
        if self.rotated_image is not None:
            return self.rotated_image
        return self.original_image
    
    def get_current_shape(self) -> Tuple[int, int]:
        """Get shape of current working image as (height, width)."""
        return self.get_current_image().shape[:2]
    
    def transform_coords_to_original(self, coords: Dict) -> Dict:
        """
        Transform coordinates from current (cropped) space to original image space.
        
        Reverses: crop offset → inverse rotation
        
        Args:
            coords: Coordinate dict with 'x', 'y', 'width', 'height' 
                    or 'left', 'top', 'right', 'bottom'
        
        Returns:
            Coordinate dict in original image space
        """
        result = coords.copy()
        
        # Normalize to left/top/right/bottom format
        if 'x' in result and 'left' not in result:
            result['left'] = result['x']
            result['top'] = result['y']
            result['right'] = result['x'] + result.get('width', 0)
            result['bottom'] = result['y'] + result.get('height', 0)
        elif 'left' in result and 'x' not in result:
            result['x'] = result['left']
            result['y'] = result['top']
            result['width'] = result.get('right', result['left']) - result['left']
            result['height'] = result.get('bottom', result['top']) - result['top']
        
        # Step 1: Undo crop (add offset to get rotated space coords)
        if self.crop_offset is not None:
            ox, oy = self.crop_offset
            result['left'] = result.get('left', 0) + ox
            result['top'] = result.get('top', 0) + oy
            result['right'] = result.get('right', 0) + ox
            result['bottom'] = result.get('bottom', 0) + oy
            if 'x' in result:
                result['x'] = result['x'] + ox
                result['y'] = result['y'] + oy
        
        # Step 2: Undo rotation (transform all 4 corners)
        if self.rotation_applied and self.inverse_rotation_matrix is not None:
            # Transform all 4 corners of the bbox
            corners = np.array([
                [[result['left'], result['top']]],
                [[result['right'], result['top']]],
                [[result['right'], result['bottom']]],
                [[result['left'], result['bottom']]]
            ], dtype=np.float32)
            
            transformed = cv2.transform(corners, self.inverse_rotation_matrix)
            
            # Get axis-aligned bounding box of transformed corners
            x_coords = transformed[:, 0, 0]
            y_coords = transformed[:, 0, 1]
            
            result['left'] = int(np.min(x_coords))
            result['top'] = int(np.min(y_coords))
            result['right'] = int(np.max(x_coords))
            result['bottom'] = int(np.max(y_coords))
        
        # Update width/height and x/y
        result['width'] = result['right'] - result['left']
        result['height'] = result['bottom'] - result['top']
        result['x'] = result['left']
        result['y'] = result['top']
        
        return result
    
    def transform_coords_to_rotated(self, coords: Dict) -> Dict:
        """
        Transform coordinates from current (cropped) space to rotated image space.
        
        Just undoes the crop offset (no rotation transform).
        
        Args:
            coords: Coordinate dict in cropped space (supports x1/y1/x2/y2 or x/y/width/height)
        
        Returns:
            Coordinate dict in rotated image space
        """
        result = coords.copy()
        
        # Normalize to left/top/right/bottom format
        if 'x1' in result:
            result['left'] = result['x1']
            result['top'] = result['y1']
            result['right'] = result['x2']
            result['bottom'] = result['y2']
        elif 'x' in result and 'left' not in result:
            result['left'] = result['x']
            result['top'] = result['y']
            result['right'] = result['x'] + result.get('width', 0)
            result['bottom'] = result['y'] + result.get('height', 0)
        
        # Add crop offset
        if self.crop_offset is not None:
            ox, oy = self.crop_offset
            result['left'] = result.get('left', 0) + ox
            result['top'] = result.get('top', 0) + oy
            result['right'] = result.get('right', 0) + ox
            result['bottom'] = result.get('bottom', 0) + oy
        
        # Update all formats
        result['x1'] = result['left']
        result['y1'] = result['top']
        result['x2'] = result['right']
        result['y2'] = result['bottom']
        result['width'] = result['right'] - result['left']
        result['height'] = result['bottom'] - result['top']
        result['x'] = result['left']
        result['y'] = result['top']
        
        return result
    
    def transform_coords_original_to_rotated(self, coords: Dict) -> Dict:
        """
        Transform coordinates from original image space to rotated image space.
        
        Uses the forward rotation matrix to transform bbox corners.
        
        Args:
            coords: Coordinate dict with 'x1', 'y1', 'x2', 'y2' or 'left', 'top', 'right', 'bottom'
        
        Returns:
            Coordinate dict in rotated image space (axis-aligned bbox around transformed corners)
        """
        result = coords.copy()
        
        # Normalize to left/top/right/bottom format
        if 'x1' in result:
            result['left'] = result['x1']
            result['top'] = result['y1']
            result['right'] = result['x2']
            result['bottom'] = result['y2']
        elif 'x' in result and 'left' not in result:
            result['left'] = result['x']
            result['top'] = result['y']
            result['right'] = result['x'] + result.get('width', 0)
            result['bottom'] = result['y'] + result.get('height', 0)
        
        # Apply rotation if applicable
        if self.rotation_applied and self.rotation_matrix is not None:
            # Transform all 4 corners of the bbox
            corners = np.array([
                [[result['left'], result['top']]],
                [[result['right'], result['top']]],
                [[result['right'], result['bottom']]],
                [[result['left'], result['bottom']]]
            ], dtype=np.float32)
            
            transformed = cv2.transform(corners, self.rotation_matrix)
            
            # Get axis-aligned bounding box of transformed corners
            x_coords = transformed[:, 0, 0]
            y_coords = transformed[:, 0, 1]
            
            result['left'] = int(np.min(x_coords))
            result['top'] = int(np.min(y_coords))
            result['right'] = int(np.max(x_coords))
            result['bottom'] = int(np.max(y_coords))
        
        # Update all formats
        result['x1'] = result['left']
        result['y1'] = result['top']
        result['x2'] = result['right']
        result['y2'] = result['bottom']
        result['width'] = result['right'] - result['left']
        result['height'] = result['bottom'] - result['top']
        result['x'] = result['left']
        result['y'] = result['top']
        
        return result
    
    def set_metadata(self, detection_method: str, confidence: float) -> None:
        """Set detection metadata."""
        self.detection_method = detection_method
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        """
        Convert context to dictionary for serialization/debugging.
        
        Returns:
            Dictionary representation of context state
        """
        return {
            'rotation_applied': bool(self.rotation_applied),
            'rotation_angle': float(self.rotation_angle),
            'original_shape': tuple(self.original_shape) if self.original_shape else None,
            'rotated_shape': tuple(self.rotated_shape) if self.rotated_shape else None,
            'crop_offset': tuple(self.crop_offset) if self.crop_offset else None,
            'crop_shape': tuple(self.crop_shape) if self.crop_shape else None,
            'detection_method': str(self.detection_method),
            'confidence': float(self.confidence),
            'has_rotated_image': bool(self.rotated_image is not None),
            'has_cropped_image': bool(self.cropped_image is not None),
            'current_image_shape': tuple(self.get_current_shape())
        }
