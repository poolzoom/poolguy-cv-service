# PoolGuy.AI CV Service: Bottle Reference Analysis

## Overview

This document outlines the **new endpoints** needed in the Python OpenCV microservice to support the **Test Strip Reference Creator** feature. Currently, bottle label analysis uses OpenAI's GPT-4 Vision API (expensive ~$0.01/call), and we want to move this functionality to the CV service for better performance and cost efficiency.

---

## Current vs Proposed Architecture

### Current (Mixed)
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Laravel OMS Application                          │
├─────────────────────────────────────────────────────────────────────┤
│  BottleLabelAnalyzerService.php  ──────► OpenAI GPT-4 Vision API   │
│  (Uses OpenAI for bottle detection)       (Expensive, ~$0.01/call) │
│                                                                      │
│  TestStripCvService.php  ──────────────► Python CV Microservice    │
│  (Uses CV for test strip analysis)        (Fast, free)              │
└─────────────────────────────────────────────────────────────────────┘
```

### Proposed (All CV)
```
┌─────────────────────────────────────────────────────────────────────┐
│                    Laravel OMS Application                          │
├─────────────────────────────────────────────────────────────────────┤
│  TestStripCvService.php  ──────────────► Python CV Microservice    │
│  (All CV operations)                                                 │
│    ├── /health (EXISTING ✅)                                        │
│    ├── /extract-colors (EXISTING ✅)                                │
│    ├── /extract-colors-at-locations (EXISTING ✅)                   │
│    ├── /validate-image-quality (EXISTING ✅)                        │
│    ├── /analyze-bottle-label (NEW 🆕)                               │
│    ├── /detect-color-swatches (NEW 🆕)                              │
│    └── /apply-transforms (NEW 🆕)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## EXISTING ENDPOINTS (Already Implemented ✅)

### ✅ POST `/extract-colors-at-locations`

**Status: WORKING** - Used to extract colors from specific pixel locations.

```json
// Request
{
    "image_path": "https://your-s3-bucket.com/image.jpg",
    "locations": [
        {"x": 100, "y": 50, "width": 40, "height": 30},
        {"x": 100, "y": 100, "width": 40, "height": 30}
    ]
}

// Response
{
    "success": true,
    "data": {
        "colors": [
            {
                "lab": {"L": 65.5, "a": -12.3, "b": 45.2},
                "rgb": {"r": 164, "g": 198, "b": 57},
                "hex": "#A4C639",
                "region": {"x": 100, "y": 50, "width": 40, "height": 30}
            }
        ],
        "processing_time_ms": 150
    }
}
```

**Current Behavior:**
- Extracts colors from center 60% of each region (avoids edge artifacts)
- Colors returned in same order as input locations
- Supports S3 signed URLs and local file paths

### ✅ POST `/extract-colors`
Automatic pad detection and color extraction from test strip images.

### ✅ POST `/validate-image-quality`
Validates image quality before processing.

### ✅ GET `/health`
Health check endpoint.

---

## NEW ENDPOINTS REQUIRED 🆕

### 1. POST `/analyze-bottle-label` (HIGH PRIORITY)

**Purpose**: Analyze a bottle label image to detect if it's a test strip product, identify pads/parameters, and locate color swatches. This replaces the OpenAI GPT-4 Vision call.

**When Called**: 
- Automatically when user uploads a bottle label image
- After user rotates/crops the image (to re-analyze)

**Request:**
```json
{
    "image_path": "https://...",
    "rotation": 0,
    "crop": {
        "x": 0,
        "y": 0, 
        "width": 800,
        "height": 600
    }
}
```

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `image_path` | Yes | string | URL or S3 signed URL to image |
| `rotation` | No | float | Degrees to rotate (-180 to 180), default 0 |
| `crop` | No | object | Crop region to apply before analysis |

**Response:**
```json
{
    "success": true,
    "data": {
        "is_test_strip_bottle": true,
        "confidence": 0.95,
        "brand_name": "AquaChek 7-Way",
        "manufacturer": "Hach Company",
        "pads": [
            {
                "parameter": "chlorine",
                "display_name": "Free Chlorine",
                "unit": "ppm",
                "detected_swatches": 6,
                "swatch_regions": [
                    {"x": 100, "y": 50, "width": 40, "height": 30},
                    {"x": 150, "y": 50, "width": 40, "height": 30},
                    {"x": 200, "y": 50, "width": 40, "height": 30},
                    {"x": 250, "y": 50, "width": 40, "height": 30},
                    {"x": 300, "y": 50, "width": 40, "height": 30},
                    {"x": 350, "y": 50, "width": 40, "height": 30}
                ],
                "estimated_values": [0, 0.5, 1, 3, 5, 10],
                "min_value": 0,
                "max_value": 10
            },
            {
                "parameter": "ph",
                "display_name": "pH",
                "unit": "pH",
                "detected_swatches": 5,
                "swatch_regions": [...],
                "estimated_values": [6.2, 6.8, 7.2, 7.6, 8.4],
                "min_value": 6.2,
                "max_value": 8.4
            }
        ],
        "text_detected": {
            "brand": "AquaChek",
            "product_line": "7-Way",
            "parameters_text": ["FREE CHLORINE", "pH", "TOTAL ALKALINITY"]
        },
        "processing_time_ms": 450
    }
}
```

**Implementation Steps:**

1. **Image Preprocessing**
   - Download image from URL
   - Apply rotation if `rotation != 0`
   - Apply crop if `crop` is provided
   - Resize if image is too large (max 2000px on longest side)

2. **Text Detection (OCR)**
   - Use Tesseract or EasyOCR
   - Look for brand names, product names
   - Look for parameter labels (FREE CHLORINE, pH, ALKALINITY, etc.)
   - Look for numeric values near color squares

3. **Color Chart Detection**
   - Convert to HSV or LAB color space
   - Detect rectangular/square colored regions using contour detection
   - Filter by:
     - Aspect ratio (roughly square, 0.7-1.3)
     - Size (reasonable swatch size, 20-100px typically)
     - Regular spacing (swatches are usually evenly spaced)
   - Group swatches into rows by Y-coordinate
   - Sort each row by X-coordinate (left to right)

4. **Parameter Matching**
   - Match detected text to known parameter keys:

   | Detected Text | Parameter Key |
   |---------------|---------------|
   | FREE CHLORINE, CHLORINE, FCL, CL | `chlorine` |
   | TOTAL CHLORINE, TCL | `total_chlorine` |
   | pH | `ph` |
   | ALKALINITY, TOTAL ALKALINITY, ALK | `alkalinity` |
   | HARDNESS, TOTAL HARDNESS | `hardness` |
   | CALCIUM, CALCIUM HARDNESS | `calcium` |
   | CYA, CYANURIC ACID, STABILIZER | `cya` |
   | BROMINE, BR | `bromine` |

5. **Value Estimation**
   - Read numeric labels near swatches using OCR
   - If not readable, use typical ranges for the parameter

**Error Response:**
```json
{
    "success": false,
    "error": "Could not detect test strip bottle in image",
    "error_code": "NO_BOTTLE_DETECTED"
}
```

---

### 2. POST `/detect-color-swatches` (MEDIUM PRIORITY)

**Purpose**: Automatically detect color swatch regions on a bottle label. More focused than `/analyze-bottle-label` - just finds the colored squares, doesn't do OCR.

**When Called**:
- When user wants to auto-detect swatch positions after adjusting image
- As a fallback if OCR fails but we can still find color regions

**Request:**
```json
{
    "image_path": "https://...",
    "rotation": 15.5,
    "crop": {"x": 50, "y": 100, "width": 600, "height": 400},
    "expected_rows": 3,
    "min_swatch_size": 20,
    "max_swatch_size": 80
}
```

| Parameter | Required | Type | Description |
|-----------|----------|------|-------------|
| `image_path` | Yes | string | URL to image |
| `rotation` | No | float | Rotation to apply first |
| `crop` | No | object | Crop to apply first |
| `expected_rows` | No | int | Hint for number of pad rows (helps grouping) |
| `min_swatch_size` | No | int | Minimum swatch dimension in pixels (default 20) |
| `max_swatch_size` | No | int | Maximum swatch dimension in pixels (default 100) |

**Response:**
```json
{
    "success": true,
    "data": {
        "detected_rows": [
            {
                "row_index": 0,
                "y_center": 105,
                "swatches": [
                    {
                        "x": 100,
                        "y": 90,
                        "width": 35,
                        "height": 30,
                        "color": {
                            "lab": {"L": 85.2, "a": -5.1, "b": 12.3},
                            "hex": "#E8F0D0",
                            "rgb": {"r": 232, "g": 240, "b": 208}
                        }
                    },
                    {
                        "x": 145,
                        "y": 90,
                        "width": 35,
                        "height": 30,
                        "color": {
                            "lab": {"L": 75.2, "a": 2.1, "b": 45.3},
                            "hex": "#D4C878",
                            "rgb": {"r": 212, "g": 200, "b": 120}
                        }
                    }
                ]
            },
            {
                "row_index": 1,
                "y_center": 180,
                "swatches": [...]
            }
        ],
        "total_swatches": 15,
        "processing_time_ms": 320
    }
}
```

**Implementation:**
1. Apply rotation/crop transforms
2. Convert to grayscale
3. Apply adaptive thresholding
4. Find contours
5. Filter contours by size and aspect ratio
6. Cluster by Y-coordinate into rows
7. Sort each row by X-coordinate
8. Extract color from center of each detected region

---

### 3. POST `/apply-transforms` (LOW PRIORITY)

**Purpose**: Apply rotation and crop to an image and return the transformed image. Useful when we want to save the user's adjustments before final extraction.

**Request:**
```json
{
    "image_path": "https://...",
    "rotation": 15.5,
    "crop": {
        "x": 50,
        "y": 100,
        "width": 600,
        "height": 400
    },
    "output_format": "jpeg",
    "quality": 90
}
```

**Response:**
```json
{
    "success": true,
    "data": {
        "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
        "mime_type": "image/jpeg",
        "dimensions": {
            "width": 600,
            "height": 400
        },
        "processing_time_ms": 85
    }
}
```

**Alternative**: Return S3 URL instead of base64 if image is uploaded to S3.

---

### 4. ENHANCEMENT: `/extract-colors-at-locations` 

**Status**: EXISTING endpoint, needs enhancement

**Current Limitation**: Doesn't support rotation/crop transforms

**Proposed Enhancement**: Add optional `rotation` and `crop` parameters

```json
{
    "image_path": "https://...",
    "locations": [...],
    "rotation": 15.5,        // NEW - optional
    "crop": {                // NEW - optional
        "x": 50,
        "y": 100,
        "width": 600,
        "height": 400
    }
}
```

If `rotation` or `crop` are provided, apply transforms BEFORE extracting colors. Location coordinates should be relative to the transformed image.

---

## User Workflow Integration

### Creating a Custom Reference

```
1. User uploads bottle label image
   ↓
2. Frontend calls POST /analyze-bottle-label
   ↓
3. CV returns: is_test_strip_bottle, detected pads, auto-detected swatches
   ↓
4. User adjusts rotation/crop in UI
   ↓
5. Frontend calls POST /analyze-bottle-label (with rotation/crop)
   OR calls POST /detect-color-swatches
   ↓
6. CV returns updated swatch positions based on transformed image
   ↓
7. User reviews/adjusts swatch positions manually (click on image)
   ↓
8. User clicks "Extract Colors"
   ↓
9. Frontend calls POST /extract-colors-at-locations 
   (with final swatch positions + rotation/crop)
   ↓
10. CV extracts LAB colors from each swatch position
    ↓
11. Frontend displays extracted colors for review
    ↓
12. User clicks "Save Reference"
    ↓
13. Laravel saves reference to database (test_strip_brands, test_strip_pads, 
    test_strip_color_references tables)
```

---

## What OpenAI Currently Does (To Be Replaced)

The current `BottleLabelAnalyzerService.php` sends this prompt to GPT-4 Vision:

```
Analyze this image of a water test strip bottle or color reference chart.

Determine:
1. Is this a water test strip bottle/container with a color reference chart?
2. What brand/product name is visible?
3. What manufacturer is shown?
4. What test parameters (pads) are shown on the color chart?
5. For each pad, how many color swatches are there and what values do they represent?

For each pad, identify:
- The parameter name (e.g., "Free Chlorine", "pH", "Alkalinity", ...)
- The unit of measurement (ppm, pH, etc.)
- The number of color swatches
- The values for each swatch (if readable)

Parameter keys should be one of: ph, chlorine, total_chlorine, bromine, alkalinity, calcium, hardness, cya
```

**The CV service should replicate this using:**
- OCR (Tesseract/EasyOCR) for text detection
- Contour detection for finding color squares
- Color analysis for grouping and extraction

---

## Implementation Priority

| Priority | Endpoint | Reason |
|----------|----------|--------|
| 🔴 HIGH | `/analyze-bottle-label` | Core feature, replaces expensive OpenAI calls |
| 🟡 MEDIUM | `/detect-color-swatches` | Useful for auto-detection after transforms |
| 🟡 MEDIUM | Enhancement to `/extract-colors-at-locations` | Add rotation/crop support |
| 🟢 LOW | `/apply-transforms` | Nice to have, can work around with existing endpoints |

---

## Error Codes

| Code | Description |
|------|-------------|
| `IMAGE_LOAD_ERROR` | Failed to load/download image |
| `IMAGE_FORMAT_ERROR` | Unsupported image format |
| `NO_BOTTLE_DETECTED` | Image doesn't appear to be a test strip bottle |
| `NO_SWATCHES_DETECTED` | Could not detect color swatches |
| `OCR_ERROR` | Text detection failed |
| `TRANSFORM_ERROR` | Failed to apply rotation/crop |
| `INVALID_PARAMETER` | Invalid request parameter |
| `INTERNAL_ERROR` | Unexpected server error |

---

## Testing Checklist

### Test Images
- [ ] AquaChek 7-Way bottle label
- [ ] AquaChek 6-in-1 bottle label  
- [ ] Taylor test strip bottle
- [ ] Generic 3-way test strip
- [ ] Rotated image (15°, 45°, 90°)
- [ ] Cropped image
- [ ] Low quality/blurry image
- [ ] Image with glare/reflection

### Test Cases
- [ ] Bottle detection accuracy (is_test_strip_bottle)
- [ ] Brand name extraction via OCR
- [ ] Pad/parameter detection accuracy
- [ ] Swatch count detection accuracy
- [ ] Swatch region coordinates accuracy
- [ ] OCR value extraction accuracy
- [ ] Rotation handling (-180° to +180°)
- [ ] Crop handling
- [ ] Combined rotation + crop
- [ ] Error handling for invalid images

---

## Questions for CV Developer

1. **OCR Library**: Which do you prefer? Tesseract, EasyOCR, or PaddleOCR?
2. **Image Storage**: Should transformed images be returned as base64 or uploaded to S3?
3. **Processing Time**: What's the expected time for `/analyze-bottle-label`?
4. **Rotation Center**: Should rotation be around image center or a specific point?
5. **Coordinate System**: For crop, is (0,0) top-left? Are coordinates in original or rotated space?

---

## Laravel Changes (After CV Implementation)

Once CV endpoints are ready, we will:

1. Add new methods to `TestStripCvService.php`:
   ```php
   public function analyzeBottleLabel(string $imagePath, float $rotation = 0, ?array $crop = null): array
   public function detectColorSwatches(string $imagePath, float $rotation = 0, ?array $crop = null, array $options = []): array
   ```

2. Update `BottleLabelAnalyzerService.php` to use CV service instead of OpenAI

3. Update frontend to re-analyze after rotation/crop changes

4. Add optional OpenAI fallback if CV confidence is low
