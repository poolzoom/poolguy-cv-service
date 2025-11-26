"""
Visual analysis script for test strip images.
Creates annotated images showing detected pads.
"""

import cv2
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.color_extraction import ColorExtractionService
from utils.image_loader import load_image

def visualize_pads(image_path, expected_pad_count=6, output_path=None):
    """Visualize detected pads on image."""
    # Load image
    image = load_image(image_path)
    h, w = image.shape[:2]
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Try to extract colors
    service = ColorExtractionService()
    result = service.extract_colors(image_path, expected_pad_count=expected_pad_count)
    
    if not result.get('success'):
        print(f"Failed to extract colors: {result.get('error')}")
        return None
    
    # Get pad regions (we need to detect them again for visualization)
    # For now, let's use a simpler approach - detect pads and draw them
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and draw contours
    min_area = (w * h) * 0.01
    max_area = (w * h) * 0.3
    
    pad_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, rw, rh = cv2.boundingRect(contour)
            aspect_ratio = rw / rh if rh > 0 else 0
            if 0.5 < aspect_ratio < 3.0:
                pad_regions.append((x, y, rw, rh, area))
    
    # Sort by x-coordinate
    pad_regions.sort(key=lambda p: p[0])
    pad_regions = pad_regions[:expected_pad_count]
    
    # Draw detected pads
    for idx, (x, y, rw, rh, area) in enumerate(pad_regions):
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + rw, y + rh), (0, 255, 0), 3)
        # Draw pad number
        cv2.putText(vis_image, f'Pad {idx}', (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add info text
    data = result.get('data', {})
    overall_conf = data.get('overall_confidence', 0)
    pad_count = len(data.get('pads', []))
    
    info_text = f'Detected: {pad_count} pads, Confidence: {overall_conf:.3f}'
    cv2.putText(vis_image, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return vis_image

if __name__ == '__main__':
    fixtures_dir = 'tests/fixtures'
    output_dir = 'tests/fixtures/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    images = [f for f in os.listdir(fixtures_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
              and not f.startswith('.')]
    
    print("Creating visualizations...")
    for img_name in sorted(images):
        img_path = os.path.join(fixtures_dir, img_name)
        print(f"Processing: {img_name}")
        
        # Try different pad counts
        for pad_count in [4, 5, 6, 7]:
            vis = visualize_pads(img_path, expected_pad_count=pad_count)
            if vis is not None:
                output_name = f"{os.path.splitext(img_name)[0]}_{pad_count}pads.jpg"
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, vis)
                print(f"  Saved: {output_name}")
                break
    
    print(f"\nVisualizations saved to: {output_dir}")



