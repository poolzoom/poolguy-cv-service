"""
YOLO training script for test strip detection.

Trains a YOLO model using a Roboflow-exported dataset.
Supports YOLOv8 and YOLOv11 models.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics package is not installed.")
    print("Install it with: pip install ultralytics")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset(data_yaml: str) -> bool:
    """
    Validate that the dataset YAML file exists and is readable.
    
    Args:
        data_yaml: Path to dataset YAML file
    
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(data_yaml):
        logger.error(f"Dataset YAML file not found: {data_yaml}")
        return False
    
    # Try to read and parse the YAML
    try:
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['path', 'train', 'val']
        for key in required_keys:
            if key not in data:
                logger.error(f"Dataset YAML missing required key: {key}")
                return False
        
        logger.info(f"Dataset YAML validated: {data_yaml}")
        logger.info(f"  Train images: {data.get('train', 'N/A')}")
        logger.info(f"  Val images: {data.get('val', 'N/A')}")
        if 'nc' in data:
            logger.info(f"  Classes: {data['nc']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to parse dataset YAML: {e}")
        return False


def train_yolo(
    data_yaml: str,
    model_name: str = 'yolov8n.pt',
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
    output_dir: str = './models'
) -> bool:
    """
    Train YOLO model on test strip detection dataset.
    
    Args:
        data_yaml: Path to dataset YAML file (Roboflow export format)
        model_name: YOLO model to use (yolov8n.pt, yolov11n.pt, etc.)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size for training
        output_dir: Directory to save trained model
    
    Returns:
        True if training succeeded, False otherwise
    """
    # Validate dataset
    if not validate_dataset(data_yaml):
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    logger.info(f"Loading model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        return False
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=output_dir,
            name='yolo_strip_detection',
            exist_ok=True,
            save=True,
            verbose=True
        )
        
        # Model will be saved to {output_dir}/yolo_strip_detection/weights/best.pt
        # Copy to {output_dir}/best.pt for easier access
        best_model_path = os.path.join(output_dir, 'yolo_strip_detection', 'weights', 'best.pt')
        final_model_path = os.path.join(output_dir, 'best.pt')
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Trained model saved to: {final_model_path}")
        else:
            logger.warning(f"Expected model file not found at {best_model_path}")
            logger.info("Model may be saved in a different location by Ultralytics")
        
        # Print evaluation metrics
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info(f"Metrics: {metrics}")
        else:
            logger.info("Training completed. Check the output directory for detailed metrics.")
        
        # Print final model location
        logger.info(f"\nFinal model location: {final_model_path}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train YOLO model for test strip detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python training/train_yolo.py --data ./dataset/data.yaml
  
  # Train with custom parameters
  python training/train_yolo.py --data ./dataset/data.yaml --epochs 100 --model yolov8s.pt
  
  # Train YOLOv11
  python training/train_yolo.py --data ./dataset/data.yaml --model yolov11n.pt
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file (Roboflow export format)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use (default: yolov8n.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt, yolov11n.pt, etc.'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training (default: 640)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./models',
        help='Output directory for trained model (default: ./models)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.epochs < 1:
        logger.error("Epochs must be >= 1")
        sys.exit(1)
    
    if args.imgsz < 32 or args.imgsz > 2048:
        logger.error("Image size must be between 32 and 2048")
        sys.exit(1)
    
    if args.batch < 1:
        logger.error("Batch size must be >= 1")
        sys.exit(1)
    
    # Run training
    success = train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        output_dir=args.output
    )
    
    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()




