"""
Visual detection review tool for PoolGuy CV Service.
Creates annotated images showing detected pads for human review.
"""

import cv2
import numpy as np
import os
import json
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.color_extraction import ColorExtractionService
from services.utils.image_quality import ImageQualityService
from utils.image_loader import load_image


class DetectionVisualizer:
    """Creates annotated images showing pad detection results."""
    
    def __init__(self, output_dir='tests/fixtures/reviews'):
        """Initialize visualizer."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.extraction_service = ColorExtractionService()
        self.quality_service = ImageQualityService()
    
    def visualize_image(
        self, 
        image_path: str, 
        expected_pad_count: int = 6,
        show_quality: bool = True
    ) -> dict:
        """
        Create annotated visualization of pad detection.
        
        Args:
            image_path: Path to test strip image
            expected_pad_count: Expected number of pads
            show_quality: Whether to show quality metrics
        
        Returns:
            Dictionary with visualization info and output paths
        """
        # Load image
        image = load_image(image_path)
        h, w = image.shape[:2]
        
        # Create visualization image (resize if too large for display)
        max_display_width = 1920
        if w > max_display_width:
            scale = max_display_width / w
            new_w = max_display_width
            new_h = int(h * scale)
            display_image = cv2.resize(image, (new_w, new_h))
            scale_factor = scale
        else:
            display_image = image.copy()
            scale_factor = 1.0
        
        vis_h, vis_w = display_image.shape[:2]
        
        # Extract colors
        result = self.extraction_service.extract_colors(
            image_path, 
            expected_pad_count=expected_pad_count
        )
        
        # Get quality metrics
        quality_result = None
        if show_quality:
            quality_result = self.quality_service.validate_quality(image_path)
        
        # Draw detected pads
        if result.get('success'):
            data = result.get('data', {})
            pads = data.get('pads', [])
            overall_conf = data.get('overall_confidence', 0)
            
            # We need to detect pads again to get regions for visualization
            # Use the same detection logic
            pad_regions = self._detect_pads_for_visualization(image, expected_pad_count)
            
            # Draw each pad
            for idx, pad in enumerate(pads):
                if idx < len(pad_regions):
                    x, y, rw, rh = pad_regions[idx]
                    # Scale coordinates if image was resized
                    x = int(x * scale_factor)
                    y = int(y * scale_factor)
                    rw = int(rw * scale_factor)
                    rh = int(rh * scale_factor)
                    
                    # Draw bounding box
                    color = self._get_confidence_color(pad.get('pad_detection_confidence', 0))
                    cv2.rectangle(display_image, (x, y), (x + rw, y + rh), color, 3)
                    
                    # Draw pad number and info
                    lab = pad.get('lab', {})
                    conf = pad.get('pad_detection_confidence', 0)
                    
                    # Label background
                    label_y = max(30, y - 10)
                    label_text = f"Pad {idx}"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_image, 
                                 (x, label_y - text_h - 5), 
                                 (x + text_w + 10, label_y + 5), 
                                 (0, 0, 0), -1)
                    
                    # Pad number
                    cv2.putText(display_image, label_text, (x + 5, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Info text below pad
                    info_y = y + rh + 25
                    info_lines = [
                        f"L:{lab.get('L', 0):.1f} a:{lab.get('a', 0):.1f} b:{lab.get('b', 0):.1f}",
                        f"Conf: {conf:.3f}"
                    ]
                    
                    for i, line in enumerate(info_lines):
                        line_y = info_y + (i * 20)
                        if line_y < vis_h - 10:
                            cv2.putText(display_image, line, (x, line_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add header info
            self._draw_header(display_image, image_path, overall_conf, len(pads), 
                            quality_result, vis_w)
        else:
            # Draw error message
            error_msg = result.get('error', 'Unknown error')
            cv2.putText(display_image, f"ERROR: {error_msg}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_review.jpg")
        cv2.imwrite(output_path, display_image)
        
        # Also save original size if different
        if scale_factor != 1.0:
            # Create full-size version
            full_vis = image.copy()
            pad_regions = self._detect_pads_for_visualization(image, expected_pad_count)
            
            if result.get('success'):
                pads = result.get('data', {}).get('pads', [])
                for idx, pad in enumerate(pads):
                    if idx < len(pad_regions):
                        x, y, rw, rh = pad_regions[idx]
                        color = self._get_confidence_color(pad.get('pad_detection_confidence', 0))
                        cv2.rectangle(full_vis, (x, y), (x + rw, y + rh), color, 5)
                        
                        lab = pad.get('lab', {})
                        conf = pad.get('pad_detection_confidence', 0)
                        label_text = f"Pad {idx}"
                        cv2.putText(full_vis, label_text, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            full_output_path = os.path.join(self.output_dir, f"{base_name}_review_fullsize.jpg")
            cv2.imwrite(full_output_path, full_vis)
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'result': result,
            'quality_result': quality_result
        }
    
    def _detect_pads_for_visualization(self, image: np.ndarray, expected_count: int) -> list:
        """Detect pads using same logic as service (for visualization)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        min_area = (w * h) * 0.01
        max_area = (w * h) * 0.3
        
        pad_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, rw, rh = cv2.boundingRect(contour)
                aspect_ratio = rw / rh if rh > 0 else 0
                if 0.5 < aspect_ratio < 3.0:
                    pad_candidates.append((x, y, rw, rh, area))
        
        pad_candidates.sort(key=lambda p: p[0])  # Sort by x
        
        if len(pad_candidates) > expected_count:
            # Take evenly spaced
            step = len(pad_candidates) / expected_count
            selected = []
            for i in range(expected_count):
                idx = int(i * step)
                selected.append(pad_candidates[idx])
            pad_candidates = selected
        
        pad_candidates.sort(key=lambda p: p[4], reverse=True)
        pad_candidates = pad_candidates[:expected_count]
        pad_candidates.sort(key=lambda p: p[0])
        
        return [(x, y, w, h) for x, y, w, h, _ in pad_candidates]
    
    def _get_confidence_color(self, confidence: float) -> tuple:
        """Get color based on confidence score."""
        if confidence >= 0.7:
            return (0, 255, 0)  # Green - good
        elif confidence >= 0.5:
            return (0, 255, 255)  # Yellow - moderate
        else:
            return (0, 165, 255)  # Orange - low
    
    def _draw_header(self, image: np.ndarray, image_path: str, overall_conf: float, 
                    pad_count: int, quality_result: dict, width: int):
        """Draw header information on image."""
        base_name = os.path.basename(image_path)
        
        # Header background
        header_height = 120
        cv2.rectangle(image, (0, 0), (width, header_height), (40, 40, 40), -1)
        
        y_offset = 25
        
        # Image name
        cv2.putText(image, f"Image: {base_name}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # Pad count and confidence
        cv2.putText(image, f"Detected: {pad_count} pads | Overall Confidence: {overall_conf:.3f}", 
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        # Quality metrics
        if quality_result and quality_result.get('success'):
            metrics = quality_result.get('metrics', {})
            brightness = metrics.get('brightness', 0)
            contrast = metrics.get('contrast', 0)
            focus = metrics.get('focus_score', 0)
            valid = quality_result.get('valid', False)
            
            status_color = (0, 255, 0) if valid else (0, 165, 255)
            status_text = "VALID" if valid else "INVALID"
            
            quality_text = (f"Quality: {status_text} | "
                          f"Brightness: {brightness:.3f} | "
                          f"Contrast: {contrast:.3f} | "
                          f"Focus: {focus:.3f}")
            cv2.putText(image, quality_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            y_offset += 20
            
            # Errors/warnings
            errors = quality_result.get('errors', [])
            warnings = quality_result.get('warnings', [])
            if errors or warnings:
                issues = []
                if errors:
                    issues.append(f"{len(errors)} error(s)")
                if warnings:
                    issues.append(f"{len(warnings)} warning(s)")
                issues_text = " | ".join(issues)
                cv2.putText(image, f"Issues: {issues_text}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    def process_directory(self, input_dir: str, expected_pad_count: int = 6) -> dict:
        """
        Process all images in a directory and create review visualizations.
        
        Args:
            input_dir: Directory containing test images
            expected_pad_count: Expected number of pads
        
        Returns:
            Summary dictionary
        """
        images = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                  and not f.startswith('.')]
        
        results = []
        summary = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'timestamp': datetime.now().isoformat(),
            'results': []
        }
        
        print(f"Processing {len(images)} images...")
        print(f"Output directory: {self.output_dir}\n")
        
        for img_name in sorted(images):
            img_path = os.path.join(input_dir, img_name)
            print(f"Processing: {img_name}")
            
            try:
                result = self.visualize_image(img_path, expected_pad_count)
                results.append(result)
                summary['processed'] += 1
                
                if result['result'].get('success'):
                    summary['successful'] += 1
                    pads = result['result'].get('data', {}).get('pads', [])
                    conf = result['result'].get('data', {}).get('overall_confidence', 0)
                    print(f"  ✓ Success: {len(pads)} pads, confidence: {conf:.3f}")
                    print(f"  → Saved: {os.path.basename(result['output_path'])}")
                else:
                    summary['failed'] += 1
                    error = result['result'].get('error', 'Unknown')
                    print(f"  ✗ Failed: {error}")
                
            except Exception as e:
                summary['failed'] += 1
                print(f"  ✗ Error: {str(e)}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        # Save summary JSON
        summary['results'] = results
        summary_path = os.path.join(self.output_dir, 'review_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Processed: {summary['processed']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Summary saved: {summary_path}")
        print(f"{'='*60}")
        
        return summary


def main():
    """Main function to run visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize pad detection for human review')
    parser.add_argument('--input', '-i', default='tests/fixtures',
                       help='Input directory with test images')
    parser.add_argument('--output', '-o', default='tests/fixtures/reviews',
                       help='Output directory for review images')
    parser.add_argument('--pads', '-p', type=int, default=6,
                       help='Expected number of pads (default: 6)')
    parser.add_argument('--image', default=None,
                       help='Process single image instead of directory')
    
    args = parser.parse_args()
    
    visualizer = DetectionVisualizer(output_dir=args.output)
    
    if args.image:
        # Process single image
        result = visualizer.visualize_image(args.image, args.pads)
        print(f"\nReview image saved: {result['output_path']}")
    else:
        # Process directory
        visualizer.process_directory(args.input, args.pads)


if __name__ == '__main__':
    main()



