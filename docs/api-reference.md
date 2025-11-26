# PoolGuy CV Service - API Reference

## Base URL
```
http://localhost:5000
```

## Authentication
Currently no authentication required (internal service). In production, consider API key authentication.

## Endpoints

### Health Check

Check service health and version information.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "poolguy-cv-service",
  "opencv_version": "4.8.1",
  "numpy_version": "1.24.3"
}
```

**Example:**
```bash
curl http://localhost:5000/health
```

---

### Extract Colors

Extract pad colors from test strip image.

**Endpoint:** `POST /extract-colors`

**Request Body:**
```json
{
  "image_path": "/path/to/test-strip.jpg",
  "expected_pad_count": 6,
  "normalize_white": true
}
```

**Parameters:**
- `image_path` (string, required): Path to test strip image (local path or S3 signed URL)
- `expected_pad_count` (integer, optional): Number of pads expected (4-7). Default: 6
- `normalize_white` (boolean, optional): Whether to apply white balance normalization. Default: `true`

**Success Response (200):**
```json
{
  "success": true,
  "data": {
    "pads": [
      {
        "pad_index": 0,
        "region": {
          "pad_index": 0,
          "x": 100,
          "y": 200,
          "width": 50,
          "height": 50,
          "left": 100,
          "top": 200,
          "right": 150,
          "bottom": 250
        },
        "lab": {
          "L": 50.0,
          "a": 0.0,
          "b": 0.0
        },
        "confidence": 0.95,
        "color_variance": 2.5,
        "pad_detection_confidence": 0.98
      },
      {
        "pad_index": 1,
        "region": {
          "pad_index": 1,
          "x": 100,
          "y": 260,
          "width": 50,
          "height": 50,
          "left": 100,
          "top": 260,
          "right": 150,
          "bottom": 310
        },
        "lab": {
          "L": 60.0,
          "a": 10.0,
          "b": 20.0
        },
        "confidence": 0.92,
        "color_variance": 3.1,
        "pad_detection_confidence": 0.96
      }
    ],
    "overall_confidence": 0.93,
    "processing_time_ms": 1250,
    "pad_count_mismatch": {
      "expected": 6,
      "detected": 5,
      "warning": "Expected 6 pads, detected 5"
    }
  }
}
```

**Response Fields:**
- `pads`: Array of pad color results, each containing:
  - `pad_index`: Zero-based index of the pad (0 to N-1)
  - `region`: Pad coordinates in original image space (absolute coordinates)
    - `x`, `y`: Top-left corner coordinates
    - `width`, `height`: Pad dimensions
    - `left`, `top`, `right`, `bottom`: Alternative coordinate format
  - `lab`: LAB color space values
    - `L`: Lightness (0-100)
    - `a`: Green-Red axis (-128 to 127)
    - `b`: Blue-Yellow axis (-128 to 127)
  - `confidence`: Overall confidence score (0.0-1.0), combines detection and extraction confidence
  - `color_variance`: Color variance within pad region (lower = more uniform)
  - `pad_detection_confidence`: Confidence from pad detection step (0.0-1.0)
- `overall_confidence`: Average confidence across all pads
- `processing_time_ms`: Total processing time in milliseconds
- `pad_count_mismatch` (optional): Present when detected pad count doesn't match expected count

**Note:** All coordinates in `region` are absolute (relative to original image).

**Error Response (400):**
```json
{
  "success": false,
  "error": "image_path is required",
  "error_code": "MISSING_PARAMETER"
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "Failed to load image: Invalid path",
  "error_code": "IMAGE_LOAD_ERROR"
}
```

**Error Response (500):**
```json
{
  "success": false,
  "error": "Strip detection failed",
  "error_code": "STRIP_DETECTION_FAILED"
}
```

**Error Response (500):**
```json
{
  "success": false,
  "error": "Pad detection failed - no pads detected",
  "error_code": "PAD_DETECTION_FAILED"
}
```

**Error Response (500):**
```json
{
  "success": false,
  "error": "Color extraction failed",
  "error_code": "COLOR_EXTRACTION_FAILED"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/extract-colors \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/test-strip.jpg",
    "expected_pad_count": 6,
    "normalize_white": true
  }'
```

**Pipeline Flow:**
The `/extract-colors` endpoint processes images through a three-step pipeline:
1. **Strip Detection**: Detects the test strip in the image using YOLO with PCA rotation correction
2. **Pad Detection**: Detects colored pads within the strip region using YOLO or OpenCV methods
3. **Color Extraction**: Extracts LAB color values from each detected pad with white balance normalization

---

### Validate Image Quality

Validate image quality before processing.

**Endpoint:** `POST /validate-image-quality`

**Request Body:**
```json
{
  "image_path": "/path/to/test-strip.jpg"
}
```

**Parameters:**
- `image_path` (string, required): Path to test strip image

**Success Response (200):**
```json
{
  "success": true,
  "valid": true,
  "metrics": {
    "brightness": 0.65,
    "contrast": 0.78,
    "focus_score": 0.85
  },
  "warnings": []
}
```

**Validation Failed Response (200):**
```json
{
  "success": true,
  "valid": false,
  "metrics": {
    "brightness": 0.15,
    "contrast": 0.25,
    "focus_score": 0.30
  },
  "errors": [
    {
      "type": "brightness",
      "message": "Image too dark - ensure good lighting",
      "threshold": 0.3,
      "actual": 0.15
    },
    {
      "type": "focus",
      "message": "Image too blurry - hold camera steady",
      "threshold": 0.5,
      "actual": 0.30
    }
  ],
  "recommendations": [
    "Ensure good lighting conditions",
    "Hold camera steady while taking photo",
    "Ensure test strip is in focus"
  ]
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "image_path is required",
  "error_code": "MISSING_PARAMETER"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/validate-image-quality \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/path/to/test-strip.jpg"
  }'
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `MISSING_PARAMETER` | Required parameter is missing |
| `INVALID_PARAMETER` | Parameter value is invalid |
| `IMAGE_LOAD_ERROR` | Failed to load image from path or URL |
| `IMAGE_QUALITY_ERROR` | Error during image quality validation |
| `PROCESSING_ERROR` | General processing error |
| `STRIP_DETECTION_FAILED` | Failed to detect test strip in image |
| `STRIP_CROP_FAILED` | Failed to crop strip region from image |
| `PAD_DETECTION_FAILED` | Failed to detect pads in strip region |
| `COLOR_EXTRACTION_FAILED` | Failed to extract colors from pads |
| `INTERNAL_ERROR` | Unexpected server error |

## Response Format

All responses follow this structure:

**Success:**
```json
{
  "success": true,
  "data": { ... }
}
```

**Error:**
```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE"
}
```

## Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request parameters
- `500 Internal Server Error` - Server error

## Rate Limiting

Currently no rate limiting. In production, consider implementing rate limiting based on Laravel service needs.

## Timeouts

- Request timeout: 30 seconds (recommended for Laravel client)
- Image processing timeout: 15 seconds (target)

## Image Formats

Supported formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)

Maximum file size: 10MB

## Integration Example (Laravel)

```php
use Illuminate\Support\Facades\Http;

$response = Http::timeout(30)->post('http://localhost:5000/extract-colors', [
    'image_path' => $s3ImageUrl,
    'expected_pad_count' => 6,
    'normalize_white' => true
]);

if ($response->successful() && $response->json('success')) {
    $data = $response->json('data');
    $pads = $data['pads'];
    $overallConfidence = $data['overall_confidence'];
    $processingTime = $data['processing_time_ms'];
    
    // Check for pad count mismatch
    if (isset($data['pad_count_mismatch'])) {
        // Log warning but continue processing
        \Log::warning('Pad count mismatch', $data['pad_count_mismatch']);
    }
    
    // Process pad colors
    foreach ($pads as $pad) {
        $lab = $pad['lab'];
        $confidence = $pad['confidence'];
        $region = $pad['region'];
        // Process each pad...
    }
} else {
    $error = $response->json('error');
    $errorCode = $response->json('error_code');
    // Handle error based on error_code
}
```

## Processing Details

### Strip Detection Methods

The pipeline uses multiple detection methods with automatic fallback:

1. **YOLO with PCA Rotation** (`yolo_pca`): 
   - Detects strip with YOLO
   - Uses PCA to detect rotation angle
   - Rotates image to correct orientation
   - Re-detects strip on rotated image
   - Provides best accuracy for rotated strips

2. **YOLO with Refinement** (`yolo_refined`):
   - YOLO detection followed by refinement
   - Improves bounding box accuracy

3. **YOLO Only** (`yolo`):
   - Direct YOLO detection without refinement
   - Fastest method

4. **OpenAI Vision** (`openai`):
   - Uses OpenAI Vision API for detection
   - Fallback when YOLO fails

### Coordinate System

The pipeline uses a sophisticated coordinate transformation system:

- **Original Image Space**: Input image coordinates
- **Rotated Space**: After rotation correction
- **Working Space**: Cropped strip region (where pad detection occurs)

All pad coordinates are automatically transformed back to original image space in the response.

### White Balance Normalization

When `normalize_white: true` (default):
- Detects white regions between pads
- Uses these regions as reference for color normalization
- Improves color accuracy across different lighting conditions
- Falls back to default method if white regions can't be detected





