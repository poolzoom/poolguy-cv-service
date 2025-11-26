#!/usr/bin/env python3
"""
Simple test script for the pipeline service.

Thin wrapper around PipelineService for command-line testing.
Saves results to experiments folder following the standard pattern.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.pipeline import PipelineService
from services.utils.debug import DebugContext
from utils.image_loader import load_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Test pipeline service on an image')
    parser.add_argument('image_path', help='Path to test strip image')
    parser.add_argument('--expected-pads', type=int, default=6, help='Expected number of pads (4-7, default: 6)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable white balance normalization')
    parser.add_argument('--save-results', action='store_true', help='Save results to experiments folder')
    parser.add_argument('--output-dir', type=str, default='experiments', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate image exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Testing Pipeline Service")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Expected pads: {args.expected_pads}")
    print(f"Normalize white: {not args.no_normalize}")
    print(f"{'='*70}\n")
    
    # Load image
    print("[1/3] Loading image...")
    try:
        image = load_image(str(image_path))
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        sys.exit(1)
    
    # Setup debug context if saving results
    debug = None
    if args.save_results:
        # Create timestamped run name to avoid overwriting previous results
        image_base = image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{image_base}_{timestamp}"
        debug = DebugContext(
            enabled=True,
            output_dir=args.output_dir,
            image_name=run_name,
            comparison_tag='pipeline'
        )
        # Log input image
        debug.add_step('00_input', 'Input Image', image, {
            'width': image.shape[1],
            'height': image.shape[0],
            'expected_pad_count': args.expected_pads
        })
    
    # Run pipeline
    print(f"\n[2/3] Running pipeline...")
    pipeline = PipelineService()
    
    try:
        result = pipeline.process_image(
            image=image,
            image_name=image_path.stem,
            expected_pad_count=args.expected_pads,
            normalize_white=not args.no_normalize,
            debug=debug
        )
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        logger.exception("Pipeline error")
        if debug:
            debug.save_log()
        sys.exit(1)
    
    # Print results
    print(f"\n[3/3] Results:")
    print(f"{'='*70}")
    
    if result.get('success'):
        pads = result.get('data', {}).get('pads', [])
        overall_confidence = result.get('data', {}).get('overall_confidence', 0.0)
        
        print(f"✓ SUCCESS")
        print(f"  Detected {len(pads)} pads")
        print(f"  Overall confidence: {overall_confidence:.3f}")
        print(f"\n  Pad Colors (LAB):")
        
        for i, pad in enumerate(pads):
            lab = pad.get('lab', {})
            confidence = pad.get('confidence', 0.0)
            region = pad.get('region', {})
            
            print(f"    Pad {i+1}:")
            print(f"      LAB: L={lab.get('L', 0):.1f}, a={lab.get('a', 0):.1f}, b={lab.get('b', 0):.1f}")
            print(f"      Confidence: {confidence:.3f}")
            if region:
                print(f"      Region: ({region.get('x', 0)}, {region.get('y', 0)}) "
                      f"{region.get('width', 0)}x{region.get('height', 0)}")
        
        # Save debug log if requested (pipeline already created visualizations)
        if args.save_results and debug:
            log_path = debug.save_log()
            print(f"\n✓ Results saved to: {log_path}")
            if log_path:
                # Extract run name from debug context for URL
                run_name = debug.image_name if hasattr(debug, 'image_name') else image_path.stem
                print(f"  View in browser: http://localhost:5000/review/run/{run_name}/pipeline")
        
        sys.exit(0)
    else:
        error = result.get('error', 'Unknown error')
        error_code = result.get('error_code', 'UNKNOWN')
        print(f"✗ FAILED")
        print(f"  Error: {error}")
        print(f"  Error code: {error_code}")
        
        if 'detected_count' in result:
            print(f"  Detected {result['detected_count']} pads (expected {args.expected_pads})")
        
        # Save debug log if requested (pipeline already created error visualizations)
        if args.save_results and debug:
            log_path = debug.save_log()
            print(f"\n  Results saved to: {log_path}")
            if log_path:
                # Extract run name from debug context for URL
                run_name = debug.image_name if hasattr(debug, 'image_name') else image_path.stem
                print(f"  View in browser: http://localhost:5000/review/run/{run_name}/pipeline")
        
        sys.exit(1)


if __name__ == '__main__':
    main()

