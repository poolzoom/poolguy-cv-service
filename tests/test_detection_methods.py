"""
Test script to test individual detection methods.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.image_loader import load_image
from utils.visual_logger import VisualLogger
from services.detection.methods import CannyDetector, ColorLinearDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_method(method_name: str, detector_class, image_path: str):
    """Test a single detection method."""
    print(f"\n{'='*60}")
    print(f"Testing {method_name}")
    print(f"{'='*60}")
    
    # Load image
    image = load_image(image_path)
    print(f"Image loaded: {image.shape}")
    
    # Create visual logger for this method
    visual_logger = VisualLogger()
    visual_logger.start_log(method_name.lower(), 'test_image')
    
    # Create detector
    detector = detector_class(visual_logger=visual_logger)
    
    # Run detection
    try:
        strip_region, orientation, angle = detector.detect(image)
        
        if strip_region is None:
            print(f"❌ {method_name} FAILED - No strip detected")
            return False
        else:
            print(f"✅ {method_name} SUCCEEDED")
            print(f"   Strip region: {strip_region}")
            print(f"   Orientation: {orientation}")
            print(f"   Angle: {angle:.2f}°")
            print(f"   Visual log: {visual_logger.save_log(image)}")
            return True
    except Exception as e:
        print(f"❌ {method_name} ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all detection methods."""
    if len(sys.argv) < 2:
        print("Usage: python test_detection_methods.py <image_path>")
        print("Example: python test_detection_methods.py tests/fixtures/PXL_20250427_161114135.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Test each method
    methods = [
        ('Canny', CannyDetector),
        ('Color Linear', ColorLinearDetector),
    ]
    
    results = {}
    for method_name, detector_class in methods:
        success = test_method(method_name, detector_class, image_path)
        results[method_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for method_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{method_name:20s}: {status}")


if __name__ == '__main__':
    main()



