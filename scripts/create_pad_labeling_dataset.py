#!/usr/bin/env python3
"""
Create dataset of rotated cropped strip images for pad labeling.

This script processes images through the YOLO-PCA pipeline to:
1. Detect the strip
2. Rotate the image to correct orientation
3. Crop the rotated strip region
4. Save cropped images ready for upload to Roboflow.com

Output: Simple directory with just the cropped images, ready to upload.
"""

import sys
import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.strip_detection import StripDetectionService
from services.pipeline.steps.strip_detection.pca_steps import detect_rotation_with_pca
from services.utils.image_transform_context import ImageTransformContext
from services.detection.yolo_detector import YoloDetector
from utils.image_loader import load_image
from config.pca_config import get_pca_config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_image_for_dataset(
    image_path: str,
    output_dir: Path,
    service: StripDetectionService,
    min_confidence: float = 0.3,
    min_strip_width: int = 50,
    min_strip_height: int = 200,
    padding: int = 10
) -> Optional[str]:
    """
    Process a single image and save cropped rotated strip.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save cropped images
        service: StripDetectionService instance
        min_confidence: Minimum YOLO confidence to accept
        min_strip_width: Minimum strip width to accept
        min_strip_height: Minimum strip height to accept
        padding: Padding around strip in pixels
    
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
    
    # Step 1: YOLO detection on original image
    yolo_detector = service.yolo_detector
    result1 = yolo_detector.detect_strip(image)
    
    if not result1.get('success'):
        logger.warning(f"YOLO detection failed for {image_name}")
        return None
    
    bbox1 = result1['bbox']
    conf1 = result1['confidence']
    
    if conf1 < min_confidence:
        logger.warning(f"Confidence too low ({conf1:.3f} < {min_confidence}) for {image_name}")
        return None
    
    # Step 2: PCA rotation detection
    pca_config = get_pca_config()
    rotation_angle, _ = detect_rotation_with_pca(image, bbox1, pca_config)
    
    # Step 3: Rotate entire image using transform context
    transform_context = ImageTransformContext(image)
    transform_context.apply_rotation(rotation_angle)
    rotated_image = transform_context.get_current_image()
    rotated_h, rotated_w = rotated_image.shape[:2]
    
    # Step 4: YOLO detection on rotated image
    result2 = yolo_detector.detect_strip(rotated_image)
    
    if not result2.get('success'):
        logger.warning(f"YOLO detection failed on rotated image for {image_name}, using original")
        bbox2 = bbox1
    else:
        bbox2 = result2['bbox']
    
    # Validate strip dimensions
    strip_width = bbox2['x2'] - bbox2['x1']
    strip_height = bbox2['y2'] - bbox2['y1']
    
    if strip_width < min_strip_width or strip_height < min_strip_height:
        logger.warning(
            f"Strip too small ({strip_width}x{strip_height}) for {image_name}. "
            f"Minimum: {min_strip_width}x{min_strip_height}"
        )
        return None
    
    # Step 5: Crop strip from rotated image with padding
    x1 = max(0, bbox2['x1'] - padding)
    y1 = max(0, bbox2['y1'] - padding)
    x2 = min(rotated_w, bbox2['x2'] + padding)
    y2 = min(rotated_h, bbox2['y2'] + padding)
    
    cropped_strip = rotated_image[y1:y2, x1:x2]
    
    if cropped_strip.size == 0:
        logger.warning(f"Empty crop for {image_name}")
        return None
    
    # Step 6: Save cropped image
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{image_name}.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), cropped_strip)
    
    logger.info(
        f"✓ Saved: {output_filename} "
        f"({cropped_strip.shape[1]}x{cropped_strip.shape[0]}, "
        f"rotation: {rotation_angle:.2f}°)"
    )
    
    return str(output_path)


def create_dataset(
    input_images: List[str],
    output_dir: str,
    min_confidence: float = 0.3,
    min_strip_width: int = 50,
    min_strip_height: int = 200,
    padding: int = 10
) -> Dict:
    """
    Create dataset from list of input images.
    
    Args:
        input_images: List of paths to input images
        output_dir: Directory to save cropped images
        min_confidence: Minimum YOLO confidence to accept
        min_strip_width: Minimum strip width to accept
        min_strip_height: Minimum strip height to accept
        padding: Padding around strip in pixels
    
    Returns:
        Dictionary with processing summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize service
    service = StripDetectionService(
        detection_method='yolo_pca',
        enable_visual_logging=False
    )
    
    results = {
        'total': len(input_images),
        'successful': 0,
        'failed': 0
    }
    
    logger.info(f"Processing {len(input_images)} images...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Min confidence: {min_confidence}")
    logger.info(f"Min dimensions: {min_strip_width}x{min_strip_height}")
    logger.info("=" * 60)
    
    for image_path in input_images:
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            results['failed'] += 1
            continue
        
        output_path_str = process_image_for_dataset(
            image_path,
            output_path,
            service,
            min_confidence=min_confidence,
            min_strip_width=min_strip_width,
            min_strip_height=min_strip_height,
            padding=padding
        )
        
        if output_path_str:
            results['successful'] += 1
        else:
            results['failed'] += 1
    
    logger.info("=" * 60)
    logger.info(f"Dataset creation complete!")
    logger.info(f"Successful: {results['successful']}/{results['total']}")
    logger.info(f"Failed: {results['failed']}/{results['total']}")
    success_rate = results['successful'] / results['total'] if results['total'] > 0 else 0
    logger.info(f"Success rate: {success_rate:.1%}")
    logger.info(f"Images saved to: {output_path}")
    logger.info(f"\nReady to upload to Roboflow!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Create dataset of rotated cropped strip images for pad labeling'
    )
    parser.add_argument(
        'input_images',
        nargs='+',
        help='Input image paths (can use glob patterns)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='dataset/pad_labeling',
        help='Output directory for cropped images (default: dataset/pad_labeling)'
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
    parser.add_argument(
        '--padding',
        type=int,
        default=10,
        help='Padding around strip in pixels (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns if needed
    import glob
    expanded_images = []
    for pattern in args.input_images:
        if '*' in pattern or '?' in pattern:
            expanded_images.extend(glob.glob(pattern))
        else:
            expanded_images.append(pattern)
    
    if not expanded_images:
        logger.error("No images found!")
        return 1
    
    # Remove duplicates and sort
    expanded_images = sorted(list(set(expanded_images)))
    
    logger.info(f"Found {len(expanded_images)} image(s) to process")
    
    # Create dataset
    summary = create_dataset(
        expanded_images,
        args.output_dir,
        min_confidence=args.min_confidence,
        min_strip_width=args.min_width,
        min_strip_height=args.min_height,
        padding=args.padding
    )
    
    return 0 if summary['successful'] > 0 else 1


if __name__ == '__main__':
    sys.exit(main())

