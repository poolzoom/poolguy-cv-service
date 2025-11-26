#!/usr/bin/env python3
"""
Review refinement experiment results.

Displays before/after images and metrics for visual review.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def review_experiment(experiment_path: str):
    """
    Review experiment results.
    
    Args:
        experiment_path: Path to experiment directory
    """
    exp_path = Path(experiment_path)
    
    if not exp_path.exists():
        logger.error(f"Experiment path does not exist: {experiment_path}")
        return
    
    # Load results
    results_path = exp_path / 'results.json'
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load config
    config_path = exp_path / 'config.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {results.get('experiment_name', 'Unknown')}")
    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    print(f"{'='*60}\n")
    
    # Display aggregate metrics
    if 'aggregate_metrics' in results:
        metrics = results['aggregate_metrics']
        print("Aggregate Metrics:")
        print(f"  Average Quality: {metrics.get('average_quality', 0):.3f}")
        print(f"  Average Tightness: {metrics.get('average_tightness', 0):.3f}")
        print(f"  Average BG Removal: {metrics.get('average_bg_removal', 0):.3f}")
        print(f"  Success Rate: {metrics.get('success_count', 0)}/{metrics.get('total_count', 0)}")
        print()
    
    # Display per-image results
    print(f"Image Results ({len(results.get('images', []))} images):")
    print(f"{'-'*60}")
    
    for idx, image_result in enumerate(results.get('images', []), 1):
        image_name = image_result.get('image_name', 'Unknown')
        print(f"\n{idx}. {image_name}")
        
        if 'error' in image_result:
            print(f"   ERROR: {image_result['error']}")
            continue
        
        if 'metrics' in image_result:
            metrics = image_result['metrics']
            print(f"   Quality: {metrics.get('overall_quality', 0):.3f}")
            print(f"   Tightness: {metrics.get('tightness_score', 0):.3f}")
            print(f"   BG Removal: {metrics.get('background_removal_score', 0):.3f}")
            print(f"   Size Reduction: {metrics.get('size_reduction', 0):.3f}")
        
        if 'visualization' in image_result:
            vis_path = exp_path / image_result['visualization']
            if vis_path.exists():
                print(f"   Visualization: {vis_path}")
            else:
                print(f"   Visualization: (not found)")
        
        # Show YOLO vs refined bbox
        if 'yolo_bbox' in image_result and 'refined_region' in image_result:
            yolo = image_result['yolo_bbox']
            refined = image_result['refined_region']
            yolo_area = (yolo['x2'] - yolo['x1']) * (yolo['y2'] - yolo['y1'])
            refined_area = (refined['right'] - refined['left']) * (refined['bottom'] - refined['top'])
            reduction = (1.0 - refined_area / yolo_area) * 100 if yolo_area > 0 else 0
            print(f"   YOLO bbox: {yolo['x1']},{yolo['y1']} -> {yolo['x2']},{yolo['y2']}")
            print(f"   Refined: {refined['left']},{refined['top']} -> {refined['right']},{refined['bottom']}")
            print(f"   Area reduction: {reduction:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"To view images, check: {exp_path / 'images'}")
    print(f"{'='*60}\n")


def compare_experiments(exp1_path: str, exp2_path: str):
    """
    Compare two experiments side-by-side.
    
    Args:
        exp1_path: Path to first experiment
        exp2_path: Path to second experiment
    """
    exp1 = Path(exp1_path)
    exp2 = Path(exp2_path)
    
    # Load results
    with open(exp1 / 'results.json', 'r') as f:
        results1 = json.load(f)
    with open(exp2 / 'results.json', 'r') as f:
        results2 = json.load(f)
    
    print(f"\n{'='*60}")
    print("Experiment Comparison")
    print(f"{'='*60}\n")
    
    print(f"Experiment 1: {results1.get('experiment_name', 'Unknown')}")
    print(f"Experiment 2: {results2.get('experiment_name', 'Unknown')}")
    print()
    
    # Compare aggregate metrics
    if 'aggregate_metrics' in results1 and 'aggregate_metrics' in results2:
        m1 = results1['aggregate_metrics']
        m2 = results2['aggregate_metrics']
        
        print("Aggregate Metrics Comparison:")
        print(f"  Quality:    {m1.get('average_quality', 0):.3f} vs {m2.get('average_quality', 0):.3f} "
              f"({'+' if m2.get('average_quality', 0) > m1.get('average_quality', 0) else ''}"
              f"{m2.get('average_quality', 0) - m1.get('average_quality', 0):.3f})")
        print(f"  Tightness:   {m1.get('average_tightness', 0):.3f} vs {m2.get('average_tightness', 0):.3f}")
        print(f"  BG Removal:  {m1.get('average_bg_removal', 0):.3f} vs {m2.get('average_bg_removal', 0):.3f}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Review refinement experiment results')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Path to experiment directory')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to second experiment for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments(args.experiment, args.compare)
    else:
        review_experiment(args.experiment)


if __name__ == '__main__':
    main()


