#!/usr/bin/env python3
"""
Prepare pad detection dataset for training.

Splits train data into train/val sets and fixes data.yaml paths.
"""

import os
import shutil
import yaml
from pathlib import Path
import random

def prepare_dataset(dataset_dir: str, val_split: float = 0.2):
    """
    Prepare dataset by creating train/val split and fixing data.yaml.
    
    Args:
        dataset_dir: Path to dataset directory
        val_split: Fraction of data to use for validation (default: 0.2)
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return False
    
    train_dir = dataset_path / 'train'
    if not train_dir.exists():
        print(f"ERROR: Train directory not found: {train_dir}")
        return False
    
    # Get all image files
    train_images_dir = train_dir / 'images'
    train_labels_dir = train_dir / 'labels'
    
    image_files = sorted(list(train_images_dir.glob('*.jpg')))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {train_images_dir}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)
    
    val_count = max(1, int(len(image_files) * val_split))
    val_images = image_files[:val_count]
    train_images = image_files[val_count:]
    
    print(f"Split: {len(train_images)} train, {len(val_images)} validation")
    
    # Create val directory structure
    val_dir = dataset_path / 'valid'
    val_images_dir = val_dir / 'images'
    val_labels_dir = val_dir / 'labels'
    
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Move validation images and labels
    for img_path in val_images:
        # Get corresponding label file
        label_name = img_path.stem + '.txt'
        label_path = train_labels_dir / label_name
        
        if label_path.exists():
            # Move image
            shutil.move(str(img_path), str(val_images_dir / img_path.name))
            # Move label
            shutil.move(str(label_path), str(val_labels_dir / label_name))
            print(f"  Moved to val: {img_path.name}")
        else:
            print(f"  WARNING: Label not found for {img_path.name}")
    
    # Update data.yaml with absolute paths
    data_yaml = dataset_path / 'data.yaml'
    
    dataset_abs_path = dataset_path.resolve()
    
    yaml_data = {
        'path': str(dataset_abs_path),
        'train': str(dataset_abs_path / 'train' / 'images'),
        'val': str(dataset_abs_path / 'valid' / 'images'),
        'test': str(dataset_abs_path / 'valid' / 'images'),  # Use val for test if needed
        'nc': 1,
        'names': ['pads']
    }
    
    with open(data_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"\n✓ Dataset prepared!")
    print(f"  Train images: {len(train_images)}")
    print(f"  Val images: {len(val_images)}")
    print(f"  data.yaml updated: {data_yaml}")
    
    return True

if __name__ == '__main__':
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset/strip-pads.v1i.yolov8'
    val_split = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    
    success = prepare_dataset(dataset_dir, val_split)
    sys.exit(0 if success else 1)








