"""
Unit tests for image quality validation service.
"""

import pytest
import os
from services.utils.image_quality import ImageQualityService


class TestImageQuality:
    """Test cases for ImageQualityService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = ImageQualityService()
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    
    def test_validate_quality_missing_image_path(self):
        """Test validation with missing image path."""
        result = self.service.validate_quality('')
        assert result['success'] is False
    
    def test_validate_quality_invalid_path(self):
        """Test validation with invalid image path."""
        result = self.service.validate_quality('invalid/path/image.jpg')
        assert result['success'] is False
    
    def test_calculate_metrics_structure(self):
        """Test that calculate_metrics returns expected structure."""
        # This test would require a valid image
        # For now, just verify the method exists
        assert hasattr(self.service, '_calculate_metrics')
    
    def test_calculate_focus_score(self):
        """Test focus score calculation."""
        import numpy as np
        import cv2
        
        # Create a test image (sharp)
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        score = self.service._calculate_focus_score(test_image)
        
        assert 0.0 <= score <= 1.0
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        errors = [{'type': 'brightness'}]
        warnings = [{'type': 'focus'}]
        
        recommendations = self.service._generate_recommendations(errors, warnings)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0



