"""
Unit tests for color extraction service.
"""

import pytest
import os
import cv2
import numpy as np
from services.pipeline.steps.color_extraction import ColorExtractionService
from services.interfaces import PadRegion


class TestColorExtraction:
    """Test cases for ColorExtractionService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = ColorExtractionService()
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    
    def test_extract_colors_invalid_pad_count(self):
        """Test extraction with invalid pad count."""
        result = self.service.extract_colors(image_path='test.jpg', expected_pad_count=3)
        assert result['success'] is False
        assert 'error_code' in result
        
        result = self.service.extract_colors(image_path='test.jpg', expected_pad_count=10)
        assert result['success'] is False
    
    def test_extract_colors_missing_parameters(self):
        """Test extraction with missing required parameters."""
        result = self.service.extract_colors()
        assert result['success'] is False
        assert result['error_code'] == 'INVALID_PARAMETER'
        assert 'image_path' in result['error'] or 'image' in result['error']
    
    def test_extract_colors_valid_pad_count(self):
        """Test extraction with valid pad count."""
        # This would require a valid test image
        # For now, just verify the method accepts valid counts
        for count in [4, 5, 6, 7]:
            # Will fail on image loading, but validates parameter acceptance
            result = self.service.extract_colors(image_path='test.jpg', expected_pad_count=count)
            # Should not fail on parameter validation
            assert 'error_code' not in result or result.get('error_code') != 'INVALID_PARAMETER'
    
    def test_extract_colors_with_image_array(self):
        """Test extraction with image array parameter."""
        # Create a dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Should fail on pad detection, but not on parameter validation
        result = self.service.extract_colors(
            image=dummy_image,
            expected_pad_count=6
        )
        # Should not fail on parameter validation
        assert 'error_code' not in result or result.get('error_code') != 'INVALID_PARAMETER'
    
    def test_extract_colors_with_pad_regions(self):
        """Test extraction with pre-detected pad regions."""
        # Create a dummy image
        dummy_image = np.zeros((200, 100, 3), dtype=np.uint8)
        
        # Create dummy pad regions
        pad_regions = [
            PadRegion(
                pad_index=0,
                x=10, y=10,
                width=30, height=30,
                left=10, top=10,
                right=40, bottom=40
            ),
            PadRegion(
                pad_index=1,
                x=10, y=50,
                width=30, height=30,
                left=10, top=50,
                right=40, bottom=80
            )
        ]
        
        # Should work with pad regions (will extract colors even from dummy image)
        result = self.service.extract_colors(
            image=dummy_image,
            pad_regions=pad_regions,
            expected_pad_count=2
        )
        
        # Should succeed (even if colors are all zeros)
        assert result['success'] is True
        assert len(result['data']['pads']) == 2
    
    def test_detect_pads_method_exists(self):
        """Test that pad detection method exists."""
        assert hasattr(self.service, '_detect_pads')
    
    def test_extract_pad_color_structure(self):
        """Test pad color extraction structure."""
        # This would require a valid image and region
        # For now, just verify the method exists
        assert hasattr(self.service, '_extract_pad_color')
    
    def test_calculate_pad_detection_confidence(self):
        """Test confidence calculation."""
        color_data = {
            'L': 50.0,
            'a': 0.0,
            'b': 0.0,
            'color_variance': 5.0
        }
        region = (10, 10, 50, 50)
        quality_metrics = {
            'brightness': 0.7,
            'contrast': 0.8,
            'focus_score': 0.9
        }
        
        confidence = self.service._calculate_pad_detection_confidence(
            color_data,
            region,
            quality_metrics,
            white_norm_success=True
        )
        
        assert 0.0 <= confidence <= 1.0



