# Create Pad Labeling Dataset

This script creates a dataset of rotated, cropped strip images ready for pad labeling to train a new YOLO pad detection model.

## Overview

The script processes images through the YOLO-PCA pipeline:
1. Detects the strip using YOLO
2. Uses PCA to detect rotation angle
3. Rotates the entire image to correct orientation
4. Re-detects the strip in the rotated image
5. Crops the rotated strip region
6. Saves cropped images and metadata

## Usage

### Basic Usage

```bash
python scripts/create_pad_labeling_dataset.py \
  tests/fixtures/*.jpg \
  -o dataset/pad_labeling
```

### With Options

```bash
python scripts/create_pad_labeling_dataset.py \
  tests/fixtures/*.jpg \
  -o dataset/pad_labeling \
  --min-confidence 0.4 \
  --min-width 60 \
  --min-height 250 \
  --expected-pads 6
```

### Parameters

- `input_images`: One or more image paths (supports glob patterns like `*.jpg`)
- `-o, --output-dir`: Output directory (default: `dataset/pad_labeling`)
- `--min-confidence`: Minimum YOLO confidence to accept (default: 0.3)
- `--min-width`: Minimum strip width in pixels (default: 50)
- `--min-height`: Minimum strip height in pixels (default: 200)
- `--expected-pads`: Expected number of pads (default: 6, for logging only)

## Output Structure

```
dataset/pad_labeling/
├── images/                          # Cropped strip images
│   ├── IMAGE1_cropped.jpg
│   ├── IMAGE2_cropped.jpg
│   └── ...
├── metadata/                        # Metadata JSON files
│   ├── IMAGE1_metadata.json
│   ├── IMAGE2_metadata.json
│   └── ...
└── dataset_summary.json             # Processing summary
```

## Metadata Format

Each metadata JSON file contains:

```json
{
  "original_image": "path/to/original.jpg",
  "original_size": {"width": 992, "height": 1762},
  "cropped_image": "path/to/cropped.jpg",
  "cropped_size": {"width": 126, "height": 1427},
  "rotation_angle": 0.026,
  "yolo_confidence_original": 0.585,
  "yolo_confidence_rotated": 0.589,
  "bbox_original": {"x1": 460, "y1": 216, "x2": 567, "y2": 1620},
  "bbox_rotated": {"x1": 461, "y1": 217, "x2": 567, "y2": 1624},
  "bbox_cropped": {"x1": 451, "y1": 207, "x2": 577, "y2": 1634, "width": 126, "height": 1427},
  "strip_dimensions": {"width": 106, "height": 1407},
  "pca_params": {...},
  "use_rotated_detection": true
}
```

## Next Steps

1. **Label the cropped images**: Use a tool like LabelImg, Roboflow, or CVAT to label pads in the cropped images
2. **Export in YOLO format**: Export labels as YOLO format (class_id x_center y_center width height, normalized 0-1)
3. **Train pad detection model**: Use the labeled dataset to train a new YOLO model for pad detection

## Tips

- Start with a small subset to verify the output quality
- Adjust `--min-confidence` to filter out low-quality detections
- Check the `dataset_summary.json` for success rate and statistics
- The cropped images are already rotated and ready for labeling
- All coordinates in metadata are in the rotated image space

## Example Workflow

```bash
# 1. Create dataset from test images
python scripts/create_pad_labeling_dataset.py \
  tests/fixtures/*.jpg \
  -o dataset/pad_labeling \
  --min-confidence 0.4

# 2. Review cropped images
open dataset/pad_labeling/images/

# 3. Label pads in cropped images (using LabelImg, Roboflow, etc.)
# Labels should be in YOLO format relative to cropped image

# 4. Train YOLO pad detection model
# Use the labeled dataset with YOLOv8 training script
```

