#!/usr/bin/env python3
"""
Experiment runner for strip refinement.

Runs refinement with different parameter sets and saves results for comparison.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import cv2
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.detection.yolo_detector import YoloDetector
from services.refinement import StripRefiner
from services.refinement.evaluator import RefinementEvaluator
from services.utils.debug import DebugContext
from config.refinement_config import get_config
from utils.image_loader import load_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(
    image_paths: List[str],
    config: Optional[Dict] = None,
    output_dir: str = None,
    compare_with: Optional[str] = None
) -> Dict:
    """
    Run refinement experiment on images.
    
    Args:
        image_paths: List of image paths to test
        config: Refinement configuration (uses default if None)
        output_dir: Output directory for results
        compare_with: Path to previous experiment for comparison
        
    Returns:
        Dictionary with experiment results
    """
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"experiments/refinement/experiment_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_to_save = config or get_config()
    config_path = output_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Initialize services
    yolo_detector = YoloDetector()
    refiner = StripRefiner(config_to_save)
    evaluator = RefinementEvaluator()
    
    results = {
        'experiment_name': output_path.name,
        'timestamp': datetime.now().isoformat(),
        'config': config_to_save,
        'images': []
    }
    
    for image_path in image_paths:
        logger.info(f"Processing {image_path}")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load {image_path}")
            continue
        
        image_name = Path(image_path).stem
        
        # Detect with YOLO
        yolo_result = yolo_detector.detect_strip(image)
        if not yolo_result.get('success'):
            logger.warning(f"YOLO detection failed for {image_path}")
            continue
        
        yolo_bbox = yolo_result['bbox']
        
        # Setup debug context
        debug = DebugContext(
            enabled=True,
            output_dir=str(images_dir),
            image_name=image_name
        )
        
        # Refine
        try:
            refined_strip = refiner.refine(image, yolo_bbox, debug)
            
            # Evaluate
            original_region = {
                'left': yolo_bbox['x1'],
                'top': yolo_bbox['y1'],
                'right': yolo_bbox['x2'],
                'bottom': yolo_bbox['y2']
            }
            metrics = evaluator.evaluate(original_region, refined_strip, image)
            
            # Save visualization
            vis = image.copy()
            cv2.rectangle(vis, (original_region['left'], original_region['top']),
                         (original_region['right'], original_region['bottom']), (255, 0, 0), 3)
            cv2.rectangle(vis, (refined_strip['left'], refined_strip['top']),
                         (refined_strip['right'], refined_strip['bottom']), (0, 255, 0), 3)
            cv2.putText(vis, f'Quality: {metrics["overall_quality"]:.2f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            vis_path = images_dir / f'{image_name}_comparison.jpg'
            cv2.imwrite(str(vis_path), vis)
            
            # Save debug log
            debug.save_log(vis)
            
            # Store results
            image_result = {
                'image_path': image_path,
                'image_name': image_name,
                'yolo_bbox': yolo_bbox,
                'refined_region': {
                    'left': refined_strip['left'],
                    'top': refined_strip['top'],
                    'right': refined_strip['right'],
                    'bottom': refined_strip['bottom']
                },
                'metrics': metrics,
                'visualization': str(vis_path.relative_to(output_path))
            }
            results['images'].append(image_result)
            
        except Exception as e:
            logger.error(f"Refinement failed for {image_path}: {e}", exc_info=True)
            results['images'].append({
                'image_path': image_path,
                'image_name': image_name,
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    successful = [r for r in results['images'] if 'metrics' in r]
    if successful:
        results['aggregate_metrics'] = {
            'average_quality': np.mean([r['metrics']['overall_quality'] for r in successful]),
            'average_tightness': np.mean([r['metrics']['tightness_score'] for r in successful]),
            'average_bg_removal': np.mean([r['metrics']['background_removal_score'] for r in successful]),
            'success_count': len(successful),
            'total_count': len(results['images'])
        }
    
    # Save results
    results_path = output_path / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment complete. Results saved to {output_path}")
    logger.info(f"Average quality: {results.get('aggregate_metrics', {}).get('average_quality', 0):.2f}")
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Run strip refinement experiments')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Image paths to test')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (JSON)')
    parser.add_argument('--preset', type=str, choices=['conservative', 'aggressive', 'balanced'],
                       help='Use preset configuration')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to previous experiment for comparison')
    parser.add_argument('--override', type=str, nargs='+',
                       help='Override config values (e.g., padding.pixels=8)')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.preset:
        config = get_config(args.preset)
    else:
        config = get_config()
    
    # Apply overrides
    if args.override:
        for override in args.override:
            key, value = override.split('=', 1)
            keys = key.split('.')
            # Navigate nested dict
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Set value (try to convert to appropriate type)
            try:
                if '.' in value:
                    d[keys[-1]] = float(value)
                else:
                    d[keys[-1]] = int(value)
            except ValueError:
                d[keys[-1]] = value
    
    # Run experiment
    results = run_experiment(
        image_paths=args.images,
        config=config,
        output_dir=args.output,
        compare_with=args.compare
    )
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {results['experiment_name']}")
    if 'aggregate_metrics' in results:
        print(f"Average quality: {results['aggregate_metrics']['average_quality']:.2f}")


if __name__ == '__main__':
    main()

