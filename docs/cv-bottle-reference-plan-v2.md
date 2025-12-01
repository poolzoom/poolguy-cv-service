# CV Service: Bottle Reference Analysis - Simple Stateless API

## Overview

The CV service provides **simple, stateless endpoints** for bottle label analysis. Each request includes an image URL and returns results with coordinates relative to that image. **No transform handling needed** - the Laravel app sends already-transformed images.

---

## Core Principle: One Image, One Truth

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Laravel App                           CV Service                   │
│   ───────────                           ──────────                   │
│                                                                      │
│   User uploads image                                                 │
│         │                                                            │
│         ▼                                                            │
│   [original.jpg] ──────────────────────► /analyze-bottle-label      │
│         │                                        │                   │
│         │                                        ▼                   │
│         │                               Returns: pads, swatches,    │
│         │                               coordinates (relative to    │
│         │                               original.jpg)               │
│         │                                        │                   │
│         ▼                                        ▼                   │
│   User reviews, rotates, crops          Laravel displays results    │
│         │                                                            │
│         ▼                                                            │
│   [cropped_v2.jpg] ────────────────────► /detect-color-swatches     │
│   (NEW image uploaded to S3)                     │                   │
│         │                                        ▼                   │
│         │                               Returns: swatch locations   │
│         │                               (relative to cropped_v2.jpg)│
│         │                                        │                   │
│         ▼                                        ▼                   │
│   User confirms swatch positions        Laravel displays results    │
│         │                                                            │
│         ▼                                                            │
│   [cropped_v2.jpg] + locations ────────► /extract-colors-at-locations│
│                                                  │                   │
│                                                  ▼                   │
│                                         Returns: LAB colors for     │
│                                         each location               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- CV service receives ONE image per request
- All coordinates in response are relative to THAT image
- When user rotates/crops, Laravel creates a NEW image and sends that
- CV service has no knowledge of previous images or transforms
- Simple and stateless

---

## Endpoint 1: `/analyze-bottle-label` (POST)

### Purpose
Full AI-powered analysis of a bottle label image. Detects brand, parameters, and swatch locations.

### Request

```json
{
    "image_path": "https://s3.../bottle.jpg",
    "hints": {
        "expected_pad_count": 7,
        "known_brand": "AquaChek"
    }
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `image_path` | Yes | string | URL to bottle image |
| `hints` | No | object | Optional hints to improve detection |

### Response

```json
{
    "success": true,
    "data": {
        "detection": {
            "is_test_strip_bottle": true,
            "confidence": 0.92
        },
        "brand": {
            "name": "AquaChek 7-Way",
            "confidence": 0.85,
            "detected_text": ["AquaChek", "7-Way", "Test Strips"]
        },
        "pads": [
            {
                "pad_index": 0,
                "parameter": "chlorine",
                "display_name": "Free Chlorine",
                "confidence": 0.90,
                "swatches": [
                    {
                        "swatch_index": 0,
                        "value": "0",
                        "region": {"x": 110, "y": 55, "width": 35, "height": 30},
                        "color": {
                            "lab": {"L": 85.2, "a": -5.1, "b": 12.3},
                            "hex": "#E8F0D0"
                        }
                    },
                    {
                        "swatch_index": 1,
                        "value": "0.5",
                        "region": {"x": 155, "y": 55, "width": 35, "height": 30},
                        "color": {
                            "lab": {"L": 78.5, "a": -2.3, "b": 25.6},
                            "hex": "#D4D878"
                        }
                    }
                ]
            },
            {
                "pad_index": 1,
                "parameter": "ph",
                "display_name": "pH",
                "confidence": 0.95,
                "swatches": [...]
            }
        ],
        "image_dimensions": {
            "width": 1200,
            "height": 800
        },
        "processing_time_ms": 1250
    }
}
```

### How It Works
1. Load image from URL
2. Send to OpenAI Vision to understand:
   - Brand/product name
   - Parameter names (pH, Chlorine, etc.)
   - Approximate swatch layout
   - Reference values (0, 0.5, 1, 3, 5, 10...)
3. Use OpenCV to:
   - Detect precise swatch rectangles
   - Extract actual LAB colors from each region
4. Return merged results

---

## Endpoint 2: `/detect-color-swatches` (POST)

### Purpose
CV-only swatch detection. **No OpenAI calls.** Fast and free. Use after user has adjusted the image and wants to re-detect swatches.

### Request

```json
{
    "image_path": "https://s3.../cropped_bottle.jpg",
    "options": {
        "min_swatch_size": 20,
        "max_swatch_size": 100,
        "expected_rows": 4
    }
}
```

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `image_path` | Yes | string | URL to (already cropped/rotated) image |
| `options` | No | object | Detection tuning parameters |

### Response

```json
{
    "success": true,
    "data": {
        "rows": [
            {
                "row_index": 0,
                "y_center": 75,
                "swatches": [
                    {
                        "region": {"x": 110, "y": 55, "width": 35, "height": 30},
                        "color": {
                            "lab": {"L": 85.2, "a": -5.1, "b": 12.3},
                            "hex": "#E8F0D0"
                        }
                    },
                    {
                        "region": {"x": 155, "y": 55, "width": 35, "height": 30},
                        "color": {
                            "lab": {"L": 78.5, "a": -2.3, "b": 25.6},
                            "hex": "#D4D878"
                        }
                    }
                ]
            },
            {
                "row_index": 1,
                "y_center": 125,
                "swatches": [...]
            }
        ],
        "total_swatches": 28,
        "image_dimensions": {
            "width": 600,
            "height": 400
        },
        "processing_time_ms": 180
    }
}
```

### How It Works (Pure OpenCV)
1. Load image from URL
2. Convert to grayscale
3. Apply edge detection (Canny)
4. Find contours
5. Filter by size and aspect ratio (square-ish shapes)
6. Cluster by Y-coordinate into rows
7. Sort each row by X-coordinate (left to right)
8. Extract LAB color from center of each detected region
9. Return grouped by rows

---

## Endpoint 3: `/extract-colors-at-locations` (POST) ✅ EXISTS

### Purpose
Extract colors from specific locations. Used when user has confirmed all swatch positions.

### Request

```json
{
    "image_path": "https://s3.../final_cropped.jpg",
    "locations": [
        {"x": 110, "y": 55, "width": 35, "height": 30},
        {"x": 155, "y": 55, "width": 35, "height": 30},
        {"x": 200, "y": 55, "width": 35, "height": 30}
    ]
}
```

### Response

```json
{
    "success": true,
    "data": {
        "colors": [
            {
                "lab": {"L": 85.2, "a": -5.1, "b": 12.3},
                "hex": "#E8F0D0",
                "rgb": {"r": 232, "g": 240, "b": 208},
                "region": {"x": 110, "y": 55, "width": 35, "height": 30}
            },
            {
                "lab": {"L": 78.5, "a": -2.3, "b": 25.6},
                "hex": "#D4D878",
                "rgb": {"r": 212, "g": 216, "b": 120},
                "region": {"x": 155, "y": 55, "width": 35, "height": 30}
            }
        ],
        "processing_time_ms": 85
    }
}
```

### Status
✅ **Already implemented and working!**

---

## Complete User Flow Example

### Step 1: Initial Upload

User uploads `bottle_photo.jpg`

```
Laravel → CV: POST /analyze-bottle-label
              {"image_path": "s3://.../bottle_photo.jpg"}

CV → Laravel: {
    "brand": {"name": "AquaChek 7-Way"},
    "pads": [
        {"parameter": "chlorine", "swatches": [...]},
        {"parameter": "ph", "swatches": [...]}
    ]
}
```

Laravel displays detected swatches overlaid on image.

---

### Step 2: User Adjusts Image

User rotates 15° and crops. Laravel:
1. Creates new image with rotation/crop applied
2. Uploads to S3 as `bottle_photo_v2.jpg`
3. Calls CV with new image

```
Laravel → CV: POST /detect-color-swatches
              {"image_path": "s3://.../bottle_photo_v2.jpg"}

CV → Laravel: {
    "rows": [
        {"row_index": 0, "swatches": [...]},
        {"row_index": 1, "swatches": [...]}
    ]
}
```

Laravel displays new swatch positions on the adjusted image.

---

### Step 3: User Fine-Tunes

User manually adjusts some swatch positions in the UI.
These adjustments are stored in Laravel, not sent to CV yet.

---

### Step 4: Final Extraction

User clicks "Save". Laravel sends confirmed positions.

```
Laravel → CV: POST /extract-colors-at-locations
              {
                  "image_path": "s3://.../bottle_photo_v2.jpg",
                  "locations": [
                      {"x": 110, "y": 55, "width": 35, "height": 30},
                      {"x": 155, "y": 55, "width": 35, "height": 30},
                      ...
                  ]
              }

CV → Laravel: {
    "colors": [
        {"lab": {"L": 85.2, "a": -5.1, "b": 12.3}, "hex": "#E8F0D0"},
        {"lab": {"L": 78.5, "a": -2.3, "b": 25.6}, "hex": "#D4D878"},
        ...
    ]
}
```

Laravel saves to database:
- `test_strip_brands`
- `test_strip_pads`
- `test_strip_color_references` (with LAB colors)

---

## Endpoint Summary

| Endpoint | Uses OpenAI | Purpose | When to Use |
|----------|-------------|---------|-------------|
| `/analyze-bottle-label` | ✅ Yes | Full analysis (brand, pads, values) | Initial upload |
| `/detect-color-swatches` | ❌ No | Find swatch rectangles | After user adjusts image |
| `/extract-colors-at-locations` | ❌ No | Extract colors from confirmed spots | Final save |

---

## Error Handling

### Error Response Format
```json
{
    "success": false,
    "error": "Human-readable error message",
    "error_code": "MACHINE_READABLE_CODE"
}
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `IMAGE_LOAD_ERROR` | 400 | Failed to load/download image |
| `INVALID_PARAMETER` | 400 | Invalid request parameter |
| `NO_SWATCHES_DETECTED` | 200 | Detection ran but found no swatches |
| `NO_BOTTLE_DETECTED` | 200 | Image doesn't appear to be a test strip bottle |
| `OPENAI_ERROR` | 500 | OpenAI API call failed |
| `OPENAI_UNAVAILABLE` | 503 | OpenAI service not configured |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Implementation Priority

| Priority | Task | Status |
|----------|------|--------|
| 1 | `/extract-colors-at-locations` | ✅ Done |
| 2 | `/detect-color-swatches` (CV-only) | 🔨 To Build |
| 3 | `/analyze-bottle-label` (OpenAI + CV) | 🔨 To Build |

---

## What CV Service Does vs Laravel

### CV Service:
- Loads images from URLs
- Runs OpenAI Vision (for `/analyze-bottle-label`)
- Runs OpenCV detection
- Extracts LAB/RGB/Hex colors
- Returns coordinates relative to the image received

### Laravel:
- Manages user session/state
- Creates rotated/cropped images when user adjusts
- Uploads new images to S3
- Maps CV results to database schema
- Stores user adjustments
- Handles final persistence

---

## Parameter Mapping Reference

For `/analyze-bottle-label`, the CV service will try to identify parameters and return standardized keys:

| Detected Text | `parameter` Key |
|---------------|-----------------|
| FREE CHLORINE, FCL, CL | `chlorine` |
| TOTAL CHLORINE, TCL | `total_chlorine` |
| pH | `ph` |
| ALKALINITY, ALK, TOTAL ALKALINITY | `alkalinity` |
| HARDNESS, TOTAL HARDNESS | `hardness` |
| CALCIUM, CALCIUM HARDNESS | `calcium` |
| CYA, CYANURIC ACID, STABILIZER | `cya` |
| BROMINE, BR | `bromine` |

---

## Questions for Laravel Team

1. **Image Upload Flow**: When user adjusts, does Laravel upload the new image to S3 immediately, or on-demand when calling CV?

2. **Swatch Editing UI**: How will users adjust swatch positions? Drag handles? Click to add/remove?

3. **Multiple Iterations**: Should we track version history, or is the "current" image always the only one that matters?

4. **Confidence Thresholds**: Should CV filter out low-confidence detections, or return everything and let Laravel filter?
