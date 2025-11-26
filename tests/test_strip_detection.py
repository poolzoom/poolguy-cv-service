#!/usr/bin/env python3
"""
Test script for StripDetectionService.
Visualizes detection results for all strategies.
"""

import sys
import os
import cv2
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.strip_detection import StripDetectionService
from utils.image_loader import load_image


def main():
    parser = argparse.ArgumentParser(description='Test strip detection service')
    parser.add_argument('image_path', help='Path to test image')
    parser.add_argument('--strategy', help='Strategy name (for compatibility, not used)', default='deterministic')
    parser.add_argument('--expected-pads', type=int, help='Expected number of pads', default=6)
    parser.add_argument('--no-visual-log', action='store_true', help='Disable visual logging')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI vision API for detection')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image_path}")
    image = load_image(args.image_path)
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Get image name for logging
    image_name = Path(args.image_path).name
    
    # Create service
    service = StripDetectionService(enable_visual_logging=not args.no_visual_log)
    
    # Run detection
    print(f"\nRunning deterministic pipeline")
    print("=" * 60)
    
    result = service.detect_strip(
        image,
        expected_pad_count=args.expected_pads,
        image_name=image_name,
        use_openai=args.use_openai
    )
    
    # Print results
    print("\nDetection Results:")
    print("=" * 60)
    print(f"Success: {result.get('success', False)}")
    print(f"Method: {result.get('method', 'unknown')}")
    
    if result.get('success'):
        strip_region = result.get('strip_region')
        if strip_region:
            print(f"\nStrip Region:")
            print(f"  Top: {strip_region['top']}")
            print(f"  Bottom: {strip_region['bottom']}")
            print(f"  Left: {strip_region['left']}")
            print(f"  Right: {strip_region['right']}")
            print(f"  Width: {strip_region['right'] - strip_region['left']}")
            print(f"  Height: {strip_region['bottom'] - strip_region['top']}")
        
        pads = result.get('pads', [])
        print(f"\nPads Detected: {len(pads)}")
        for pad in pads:
            print(f"  Pad {pad.get('pad_index', '?')}: "
                  f"({pad['x']}, {pad['y']}) "
                  f"{pad['width']}x{pad['height']} "
                  f"confidence={pad.get('confidence', 0.0):.2f}")
        
        print(f"\nOrientation: {result.get('orientation', 'unknown')}")
        print(f"Angle: {result.get('angle', 0.0):.1f}°")
        print(f"Handle Position: {result.get('handle_position', 'unknown')}")
        print(f"Confidence: {result.get('overall_confidence', result.get('confidence', 0.0)):.2f}")
        
        if result.get('visual_log_path'):
            print(f"\nVisual log saved to: {result['visual_log_path']}")
        
        # Create summary visualization
        vis = image.copy()
        
        # Draw strip region
        if strip_region:
            cv2.rectangle(vis,
                         (strip_region['left'], strip_region['top']),
                         (strip_region['right'], strip_region['bottom']),
                         (0, 255, 0), 3)
            cv2.putText(vis, 'Strip', (strip_region['left'], strip_region['top'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw pads
        for pad in pads:
            x, y = pad['x'], pad['y']
            w_pad, h_pad = pad['width'], pad['height']
            idx = pad.get('pad_index', 0)
            cv2.rectangle(vis, (x, y), (x + w_pad, y + h_pad), (255, 0, 0), 2)
            cv2.putText(vis, f'Pad {idx}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add text info
        info_y = 30
        method = result.get('method', 'opencv')
        cv2.putText(vis, f"Method: {method.upper()}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(vis, f"Pads: {len(pads)}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(vis, f"Confidence: {result.get('overall_confidence', result.get('confidence', 0.0)):.2f}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save visualization
        output_path = f"tests/fixtures/detection_logs/{Path(args.image_path).stem}_summary_{args.strategy}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis)
        print(f"\nSummary visualization saved to: {output_path}")
        
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        
        # Show strategy results if available
        strategy_results = result.get('strategy_results', [])
        if strategy_results:
            print("\nStrategy Results:")
            for sr in strategy_results:
                print(f"  {sr.get('strategy', 'unknown')}: "
                      f"success={sr.get('success', False)}, "
                      f"confidence={sr.get('confidence', 0.0):.2f}")
                if not sr.get('success'):
                    print(f"    Error: {sr.get('error', 'Unknown')}")


if __name__ == '__main__':
    main()
