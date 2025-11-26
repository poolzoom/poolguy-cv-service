"""
Image quality validation service for PoolGuy CV Service.
Validates brightness, contrast, and focus/blur of test strip images.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from utils.image_loader import load_image

logger = logging.getLogger(__name__)


class ImageQualityService:
    """Service for validating image quality metrics."""
    
    # Quality thresholds (configurable via environment)
    BRIGHTNESS_MIN = 0.3
    BRIGHTNESS_MAX = 0.9
    CONTRAST_MIN = 0.5
    FOCUS_MIN = 0.5
    
    def __init__(self):
        """Initialize image quality service."""
        self.logger = logging.getLogger(__name__)
    
    def validate_quality(
        self, 
        image_path: str,
        brightness_min: Optional[float] = None,
        brightness_max: Optional[float] = None,
        contrast_min: Optional[float] = None,
        focus_min: Optional[float] = None
    ) -> Dict:
        """
        Validate image quality metrics.
        
        Args:
            image_path: Path to image (local or URL)
            brightness_min: Minimum brightness threshold (0-1)
            brightness_max: Maximum brightness threshold (0-1)
            contrast_min: Minimum contrast threshold (0-1)
            focus_min: Minimum focus score threshold (0-1)
        
        Returns:
            Dictionary with validation results:
            {
                'success': bool,
                'valid': bool,
                'metrics': {...},
                'errors': [...],
                'warnings': [...],
                'recommendations': [...]
            }
        """
        try:
            # Load image
            image = load_image(image_path)
            
            # Calculate metrics
            metrics = self._calculate_metrics(image)
            
            # Use provided thresholds or defaults
            brightness_min = brightness_min or self.BRIGHTNESS_MIN
            brightness_max = brightness_max or self.BRIGHTNESS_MAX
            contrast_min = contrast_min or self.CONTRAST_MIN
            focus_min = focus_min or self.FOCUS_MIN
            
            # Validate against thresholds
            errors = []
            warnings = []
            
            brightness = metrics['brightness']
            contrast = metrics['contrast']
            focus_score = metrics['focus_score']
            
            # Check brightness
            if brightness < brightness_min:
                errors.append({
                    'type': 'brightness',
                    'message': f'Image too dark (brightness: {brightness:.2f}) - ensure good lighting',
                    'threshold': brightness_min,
                    'actual': brightness
                })
            elif brightness > brightness_max:
                errors.append({
                    'type': 'brightness',
                    'message': f'Image too bright (brightness: {brightness:.2f}) - reduce lighting',
                    'threshold': brightness_max,
                    'actual': brightness
                })
            elif brightness < brightness_min * 1.2:  # Warning zone
                warnings.append({
                    'type': 'brightness',
                    'message': f'Image may be too dark (brightness: {brightness:.2f})',
                    'threshold': brightness_min,
                    'actual': brightness
                })
            
            # Check contrast
            if contrast < contrast_min:
                errors.append({
                    'type': 'contrast',
                    'message': f'Image has low contrast ({contrast:.2f}) - ensure clear lighting',
                    'threshold': contrast_min,
                    'actual': contrast
                })
            elif contrast < contrast_min * 1.2:  # Warning zone
                warnings.append({
                    'type': 'contrast',
                    'message': f'Image may have low contrast ({contrast:.2f})',
                    'threshold': contrast_min,
                    'actual': contrast
                })
            
            # Check focus
            if focus_score < focus_min:
                errors.append({
                    'type': 'focus',
                    'message': f'Image too blurry (focus score: {focus_score:.2f}) - hold camera steady',
                    'threshold': focus_min,
                    'actual': focus_score
                })
            elif focus_score < focus_min * 1.2:  # Warning zone
                warnings.append({
                    'type': 'focus',
                    'message': f'Image may be blurry (focus score: {focus_score:.2f})',
                    'threshold': focus_min,
                    'actual': focus_score
                })
            
            # Generate recommendations
            recommendations = self._generate_recommendations(errors, warnings)
            
            # Determine if valid
            is_valid = len(errors) == 0
            
            return {
                'success': True,
                'valid': is_valid,
                'metrics': metrics,
                'errors': errors,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f'Error validating image quality: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'IMAGE_QUALITY_ERROR'
            }
    
    def _calculate_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate image quality metrics.
        
        Args:
            image: OpenCV image array in BGR format
        
        Returns:
            Dictionary with brightness, contrast, and focus_score
        """
        # Convert to grayscale for some calculations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (normalized average luminance)
        brightness = np.mean(gray) / 255.0
        
        # Calculate contrast (normalized standard deviation)
        contrast = np.std(gray) / 255.0
        
        # Calculate focus score (Laplacian variance)
        focus_score = self._calculate_focus_score(gray)
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'focus_score': float(focus_score)
        }
    
    def _calculate_focus_score(self, gray_image: np.ndarray) -> float:
        """
        Calculate focus/blur score using Laplacian variance.
        Higher values indicate sharper images.
        
        Args:
            gray_image: Grayscale image array
        
        Returns:
            Focus score (0-1, normalized)
        """
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        
        # Calculate variance
        variance = laplacian.var()
        
        # Normalize to 0-1 range (typical values: 0-500, normalize to 0-1)
        # Using sigmoid-like normalization for better distribution
        normalized = min(1.0, variance / 500.0)
        
        return normalized
    
    def _generate_recommendations(
        self, 
        errors: List[Dict], 
        warnings: List[Dict]
    ) -> List[str]:
        """
        Generate user-friendly recommendations based on errors and warnings.
        
        Args:
            errors: List of error dictionaries
            warnings: List of warning dictionaries
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        error_types = {e['type'] for e in errors}
        warning_types = {w['type'] for w in warnings}
        
        if 'brightness' in error_types:
            recommendations.append('Ensure good lighting conditions - avoid shadows and glare')
        elif 'brightness' in warning_types:
            recommendations.append('Consider improving lighting for better results')
        
        if 'contrast' in error_types:
            recommendations.append('Ensure clear, even lighting across the test strip')
        elif 'contrast' in warning_types:
            recommendations.append('Better lighting will improve contrast')
        
        if 'focus' in error_types:
            recommendations.append('Hold camera steady while taking photo')
            recommendations.append('Ensure test strip is in focus before capturing')
        elif 'focus' in warning_types:
            recommendations.append('Try to hold camera more steady for sharper image')
        
        if not recommendations:
            recommendations.append('Image quality looks good!')
        
        return recommendations



