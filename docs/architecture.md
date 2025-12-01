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

## Bottle Pipeline Architecture

The bottle pipeline processes test strip bottle images to extract pad names, reference ranges, and color mappings. It's designed to be simpler than the test strip pipeline while handling multiple images when pads wrap around the bottle.

### Pipeline Flow

```
Image Input(s)
    ↓
[Text Extraction Service]
    ├── OpenAI Vision (Primary)
    └── Tesseract OCR (Fallback)
    ↓
Pad Names + Reference Ranges
    ↓
[Pad Detection Service]
    ├── OpenAI Vision (Primary)
    ├── YOLO (If bottle model available)
    └── OpenCV Contour Detection (Fallback)
    ↓
Pad Regions (coordinates)
    ↓
[Reference Square Detection Service]
    ├── OpenCV Square Detection
    └── AI Vision Enhancement
    ↓
Reference Color Squares
    ↓
[Color Mapping Service]
    ├── Color Extraction (LAB)
    └── CIEDE2000 Matching
    ↓
Mapped Pads (with names, ranges, colors)
    ↓
[Multi-Image Handler] (if multiple images)
    ├── Overlap Detection (IoU)
    └── Result Merging
    ↓
Final Result
```

### Service Hierarchy

#### 1. Text Extraction Service (`services/pipeline/steps/bottle/text_extraction.py`)

**Purpose:** Extract pad names and reference ranges from bottle images.

**Methods:**
- **OpenAI Vision** (Primary): Uses GPT-4o vision API to extract text with context understanding
- **Tesseract OCR** (Fallback): Local OCR for text extraction

**Output:**
- List of pad names with locations
- Reference range text for each pad

#### 2. Bottle Pad Detection Service (`services/pipeline/steps/bottle/pad_detection.py`)

**Purpose:** Detect colored pads on the bottle.

**Methods:**
- **OpenAI Vision**: AI-powered pad detection
- **YOLO**: Object detection (if bottle-trained model available)
- **OpenCV**: Contour-based detection as fallback

**Output:**
- List of pad regions with coordinates and confidence

#### 3. Reference Square Detection Service (`services/pipeline/steps/bottle/reference_square_detection.py`)

**Purpose:** Detect and extract reference color squares.

**Methods:**
- **OpenCV**: Square/rectangular region detection
- **AI Vision**: Understanding square associations and values

**Output:**
- List of reference squares with LAB colors and associated values

#### 4. Color Mapping Service (`services/pipeline/steps/bottle/color_mapping.py`)

**Purpose:** Map detected pad colors to reference squares.

**Methods:**
- **CIEDE2000**: Color distance calculation
- **Color Matching**: Match pad colors to reference squares

**Output:**
- Complete pad mapping with names, ranges, colors, and mapped values

#### 5. Multi-Image Handler (`services/pipeline/bottle/multi_image_handler.py`)

**Purpose:** Handle multiple images when pads wrap around the bottle.

**Methods:**
- **IoU Calculation**: Detect overlapping pads across images
- **Result Merging**: Merge pad detections and deduplicate

**Output:**
- Unified pad map from all images

### Data Structures

#### BottlePadRegion

```python
{
    "pad_index": int,
    "name": Optional[str],  # Pad name from OCR
    "region": PadRegion,  # Pad coordinates
    "reference_range": Optional[str],  # Reference range text
    "reference_squares": List[ReferenceSquare],  # Reference color squares
    "detected_color": Optional[LabColor],  # Detected pad color
    "mapped_value": Optional[str],  # Mapped value based on reference squares
    "confidence": float  # Overall confidence (0.0-1.0)
}
```

#### ReferenceSquare

```python
{
    "color": LabColor,  # LAB color of reference square
    "value": Optional[str],  # Associated value (e.g., "7.2", "low")
    "region": PadRegion,  # Square coordinates
    "confidence": float,  # Detection confidence
    "associated_pad": Optional[int]  # Pad index this square is associated with
}
```

### Multi-Image Processing

When multiple images are provided:
1. Each image is processed independently through the pipeline
2. Overlapping pads are detected using IoU (Intersection over Union)
3. Pads with IoU >= threshold (default 0.5) are considered duplicates
4. Duplicate pads are merged, keeping the highest confidence version
5. Reference squares are deduplicated by color similarity

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

