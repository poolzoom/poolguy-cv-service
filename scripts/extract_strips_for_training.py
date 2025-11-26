#!/usr/bin/env python3
"""
Extract cropped strip images from pipeline for pad detection training.

This script processes images through the current pipeline to:
1. Detect and refine the strip (using yolo_pca method)
2. Extract the cropped/prepared strip image
3. Save cropped images ready for upload to Roboflow.com for pad labeling

Output: Directory with cropped strip images, ready to upload.
"""

import sys
import os
import cv2
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.strip_detection import StripDetectionService
from utils.image_loader import load_image

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_strip_from_image(
    image_path: str,
    output_dir: Path,
    strip_service: StripDetectionService,
    min_confidence: float = 0.3,
    min_strip_width: int = 50,
    min_strip_height: int = 200
) -> Optional[str]:
    """
    Process a single image and extract the cropped strip.
    
    Uses the current pipeline's strip detection (yolo_pca with refinement)
    to get the final cropped/prepared strip image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save cropped images
        strip_service: StripDetectionService instance
        min_confidence: Minimum YOLO confidence to accept
        min_strip_width: Minimum strip width to accept
        min_strip_height: Minimum strip height to accept
    
    Returns:
        Path to saved image, or None if failed
    """
    image_name = Path(image_path).stem
    
    logger.info(f"Processing: {image_path}")
    
    # Load image
    image = load_image(image_path)
    if image is None:
        logger.warning(f"Failed to load image: {image_path}")
        return None
    
    # Run strip detection using current pipeline method (yolo_pca with refinement)
    strip_result = strip_service.detect_strip(image, debug=None)
    
    if not strip_result.get('success'):
        error = strip_result.get('error', 'Unknown error')
        logger.warning(f"Strip detection failed for {image_name}: {error}")
        return None
    
    # Get transform context which contains the cropped/prepared strip
    transform_context = strip_result.get('transform_context')
    if transform_context is None:
        logger.warning(f"No transform context returned for {image_name}")
        return None
    
    # Get the cropped strip (prepared > cropped > rotated)
    # We want the cropped strip (before preparation/resizing) for training
    cropped_strip = transform_context.cropped_strip
    if cropped_strip is None:
        # Fallback to prepared strip if cropped not available
        cropped_strip = transform_context.prepared_strip
        if cropped_strip is None:
            logger.warning(f"No cropped strip available for {image_name}")
            return None
    
    # Validate strip dimensions
    h, w = cropped_strip.shape[:2]
    if w < min_strip_width or h < min_strip_height:
        logger.warning(
            f"Strip too small ({w}x{h}) for {image_name}. "
            f"Minimum: {min_strip_width}x{min_strip_height}"
        )
        return None
    
    # Get confidence from strip result (try multiple fields)
    confidence = (
        strip_result.get('yolo_confidence') or
        strip_result.get('confidence') or
        strip_result.get('strip_region', {}).get('confidence') or
        0.0
    )
    if confidence < min_confidence:
        logger.warning(f"Confidence too low ({confidence:.3f} < {min_confidence}) for {image_name}")
        return None
    
    # Save cropped image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{image_name}_cropped.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), cropped_strip)
    
    logger.info(
        f"✓ Saved: {output_filename} "
        f"({w}x{h}px, confidence: {confidence:.3f})"
    )
    
    return str(output_path)


def extract_strips_from_folder(
    input_folder: str,
    output_dir: str,
    min_confidence: float = 0.3,
    min_strip_width: int = 50,
    min_strip_height: int = 200
) -> Dict:
    """
    Extract cropped strips from all images in a folder.
    
    Args:
        input_folder: Folder containing input images
        output_dir: Directory to save cropped images
        min_confidence: Minimum YOLO confidence to accept
        min_strip_width: Minimum strip width to accept
        min_strip_height: Minimum strip height to accept
    
    Returns:
        Dictionary with processing summary
    """
    input_path = Path(input_folder)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix in image_extensions
    ]
    
    if not image_files:
        logger.error(f"No image files found in {input_folder}")
        return {
            'total': 0,
            'successful': 0,
            'failed': 0
        }
    
    # Initialize strip detection service
    strip_service = StripDetectionService(
        detection_method='yolo_pca',
        enable_visual_logging=False
    )
    
    results = {
        'total': len(image_files),
        'successful': 0,
        'failed': 0,
        'failed_images': []
    }
    
    logger.info(f"Processing {len(image_files)} images from {input_folder}...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Min confidence: {min_confidence}")
    logger.info(f"Min dimensions: {min_strip_width}x{min_strip_height}")
    logger.info("=" * 60)
    
    for image_file in sorted(image_files):
        output_path_str = extract_strip_from_image(
            str(image_file),
            output_path,
            strip_service,
            min_confidence=min_confidence,
            min_strip_width=min_strip_width,
            min_strip_height=min_strip_height
        )
        
        if output_path_str:
            results['successful'] += 1
        else:
            results['failed'] += 1
            results['failed_images'].append(image_file.name)
    
    logger.info("=" * 60)
    logger.info(f"Extraction complete!")
    logger.info(f"Successful: {results['successful']}/{results['total']}")
    logger.info(f"Failed: {results['failed']}/{results['total']}")
    success_rate = results['successful'] / results['total'] if results['total'] > 0 else 0
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Images saved to: {output_path}")
    
    if results['failed_images']:
        logger.warning(f"Failed images: {', '.join(results['failed_images'])}")
    
    logger.info(f"\nReady to upload to Roboflow for pad labeling!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract cropped strip images from pipeline for pad detection training'
    )
    parser.add_argument(
        'input_folder',
        help='Input folder containing images to process'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='dataset/roboflow_upload_v3',
        help='Output directory for cropped images (default: dataset/roboflow_upload_v3)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.3,
        help='Minimum YOLO confidence to accept (default: 0.3)'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=50,
        help='Minimum strip width in pixels (default: 50)'
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=200,
        help='Minimum strip height in pixels (default: 200)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        logger.error(f"Input folder not found: {args.input_folder}")
        return 1
    
    # Extract strips
    summary = extract_strips_from_folder(
        args.input_folder,
        args.output_dir,
        min_confidence=args.min_confidence,
        min_strip_width=args.min_width,
        min_strip_height=args.min_height
    )
    
    return 0 if summary['successful'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

