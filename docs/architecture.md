# PoolGuy CV Service - Architecture Documentation

## Overview

The PoolGuy CV Service is a Flask-based microservice that processes pool test strip images to detect strips, identify colored pads, and extract LAB color values for chemical analysis.

## Service Architecture

### Pipeline Flow

```
Image Input
    ↓
[Strip Detection Service]
    ├── YOLO Detector (Primary)
    ├── OpenAI Detector (Needs Refinement)
    └── OpenCV Detector (Fallback, Needs Refinement)
    ↓
Strip Region (coordinates)
    ↓
[Pad Detection Service]
    ├── Pattern-Based Detector
    ├── Color-Based Detector
    └── Edge-Based Detector
    ↓
Pad Regions (coordinates)
    ↓
[Color Extraction Service]
    ├── LAB Color Extractor
    ├── White Balance Normalizer
    └── Confidence Calculator
    ↓
Color Results (LAB values + coordinates)
```

## Service Hierarchy

### 1. Strip Detection Service (`services/strip_detection.py`)

**Purpose:** Locate the test strip in the image.

**Input:**
- Image (numpy array, BGR format)
- Optional: expected_pad_count, image_name, debug flag

**Output:**
- `DetectionResult` with `StripRegion` containing:
  - Absolute coordinates (left, top, right, bottom)
  - Dimensions (width, height)
  - Detection method used
  - Confidence score
  - Orientation (vertical/horizontal)
  - Rotation angle

**Detection Methods:**
1. **yolo_pca** (New) - YOLO → PCA rotation detection → Rotate image → YOLO re-detection
   - Uses Principal Component Analysis (PCA) on foreground pixels to detect rotation
   - Rotates entire image to correct orientation
   - Re-detects strip in rotated image for improved accuracy
   - Simplified pipeline with fewer hardcoded parameters
2. **yolo_refined** (Default) - YOLO + refinement pipeline
   - Uses YOLO for initial detection
   - Applies multi-stage refinement (orientation → projection → edges → tightening)
3. **yolo** - YOLO only (no refinement)
   - Simple YOLO detection without post-processing
4. **openai** - OpenAI Vision API
   - Uses GPT-4o vision API for detection
5. **auto** - Try methods in order until one succeeds
   - Attempts: yolo_pca → yolo_refined → yolo → openai

### 2. Pad Detection Service (`services/pad_detection.py`)

**Purpose:** Detect individual colored pads within a detected strip.

**Input:**
- Cropped strip image (numpy array)
- `StripRegion` object
- Expected pad count (4-7)

**Output:**
- `PadDetectionResult` with list of `PadRegion` objects:
  - Pad index
  - Coordinates (relative to strip image)
  - Detection confidence

**Detection Methods:**
- Pattern-based detection
- Color-based detection
- Edge-based detection
- Relative darkness detection

**Note:** Coordinates are relative to the cropped strip image and must be transformed to absolute coordinates by the caller.

### 3. Color Extraction Service (`services/color_extraction.py`)

**Purpose:** Extract LAB color values from detected pad regions.

**Input:**
- Image (full or cropped)
- List of pad regions (with coordinates)
- Expected pad count

**Output:**
- `ColorExtractionResult` with list of `ColorResult` objects:
  - Pad index
  - Pad region (coordinates)
  - LAB color values (L, a, b)
  - Detection confidence
  - Color variance

## Data Structures

### StripRegion

```python
{
    "left": int,        # Left coordinate (absolute)
    "top": int,         # Top coordinate (absolute)
    "right": int,       # Right coordinate (absolute)
    "bottom": int,      # Bottom coordinate (absolute)
    "width": int,       # Width (right - left)
    "height": int,      # Height (bottom - top)
    "confidence": float, # Detection confidence (0.0-1.0)
    "detection_method": "yolo" | "yolo_pca" | "yolo_refined" | "opencv" | "openai",
    "orientation": "vertical" | "horizontal",
    "angle": float      # Rotation angle in degrees
}
```

### PadRegion

```python
{
    "pad_index": int,   # Pad index (0-based)
    "x": int,           # Left coordinate
    "y": int,           # Top coordinate
    "width": int,       # Width
    "height": int,      # Height
    "left": int,        # Left coordinate (same as x)
    "top": int,         # Top coordinate (same as y)
    "right": int,       # Right coordinate (x + width)
    "bottom": int       # Bottom coordinate (y + height)
}
```

### ColorResult

```python
{
    "pad_index": int,
    "region": PadRegion,  # Pad coordinates
    "lab": {
        "L": float,      # Lightness (0-100)
        "a": float,      # Green-Red axis (-128 to 127)
        "b": float       # Blue-Yellow axis (-128 to 127)
    },
    "confidence": float,    # Detection confidence (0.0-1.0)
    "color_variance": float # Color variance within pad
}
```

## Coordinate System

### Absolute Coordinates

All coordinates in API responses are **absolute** (relative to the original input image).

### Coordinate Transformation

When a strip is detected and cropped:
1. Strip coordinates are absolute (in original image)
2. Pad coordinates are relative (in cropped strip image)
3. Pad coordinates must be transformed: `absolute = relative + strip_offset`

Example:
- Strip: left=100, top=200
- Pad (relative): x=10, y=20
- Pad (absolute): x=110, y=220

## Debug Mode

### Enabling Debug Mode

1. **Query Parameter:** `?debug=true` in API request
2. **Environment Variable:** `DEBUG_MODE=true`

### Debug Output

When debug mode is enabled:
- Visual logs are generated showing each step
- Debug information is included in API response:
  ```json
  {
    "debug": {
      "visual_log_path": "path/to/log",
      "enabled": true
    }
  }
  ```

## API Endpoints

### POST /detect-strip

Complete pipeline: strip detection → pad detection → color extraction.

**Response includes:**
- Strip coordinates (absolute)
- Pad coordinates (absolute)
- LAB color values
- Confidence scores
- Debug info (if enabled)

### POST /extract-colors

Extract colors from image (assumes strip already detected or uses internal detection).

**Response includes:**
- Pad coordinates (absolute)
- LAB color values
- Confidence scores

## File Organization

```
services/
  ├── interfaces.py          # Type definitions (TypedDict)
  ├── debug.py               # Debug utilities
  ├── strip_detection.py     # Strip detection orchestrator
  ├── pad_detection.py       # Pad detection service
  ├── color_extraction.py    # Color extraction service
  ├── yolo_detector.py       # YOLO detector
  ├── openai_vision.py       # OpenAI detector (needs refinement)
  └── detection_methods/     # OpenCV detectors
      ├── base_detector.py
      ├── canny_detector.py
      └── color_linear_detector.py
```

## Testing

Test files are organized in `tests/` directory:
- `tests/test_yolo_detection.py` - YOLO detection tests
- `tests/test_full_pipeline.py` - End-to-end pipeline tests
- `tests/test_strip_detection.py` - Strip detection tests

## Future Improvements

1. **Refine OpenAI Detector** - Improve reliability and accuracy
2. **Refine OpenCV Detectors** - Improve fallback detection methods
3. **Service Separation** - Further split large services into focused components
4. **Enhanced Debugging** - More detailed step-by-step visualization
5. **Performance Optimization** - Cache models, optimize image processing

