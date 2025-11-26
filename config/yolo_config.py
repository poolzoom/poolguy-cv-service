"""
YOLO configuration for strip detection.
"""

import os
from typing import Optional

# Model path - can be overridden by environment variable
MODEL_PATH: str = os.getenv('YOLO_MODEL_PATH', './models/best.pt')

# Image size for YOLO inference
IMG_SIZE: int = int(os.getenv('YOLO_IMG_SIZE', '640'))

# Confidence threshold for YOLO detections (lowered for small dataset)
CONFIDENCE_THRESHOLD: float = float(os.getenv('YOLO_CONFIDENCE_THRESHOLD', '0.01'))

