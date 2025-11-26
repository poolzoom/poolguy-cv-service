# Orientation Detection Settings Guide

## Overview

The orientation detection system has been significantly improved with configurable parameters that allow fine-tuning of the detection process. This guide explains all available settings and how they affect detection.

## Key Improvements

1. **Configurable Parameters**: All detection parameters are now configurable via environment variables or config file
2. **Multiple Rotation Combination Methods**: Choose how to combine left/right edge rotations (average, extreme, weighted)
3. **Better Visualization**: See all detected lines (gray) vs selected lines (yellow) in debug output
4. **Improved Angle Calculation**: More accurate conversion from edge angles to rotation needed

## Configuration Parameters

### Basic Settings

#### `method`
- **Default**: `'hough'`
- **Options**: `'hough'` or `'minarearect'`
- **Description**: Primary detection method. Hough line detection is recommended for most cases.

#### `max_rotation_angle`
- **Default**: `30.0` degrees
- **Description**: Maximum rotation angle to apply. Prevents over-correction.

#### `min_rotation_threshold`
- **Default**: `0.1` degrees
- **Description**: Minimum rotation angle to apply. Angles smaller than this are ignored.

#### `rotation_combination`
- **Default**: `'average'`
- **Options**: `'average'`, `'extreme'`, `'weighted'`
- **Description**: How to combine left and right edge rotations:
  - `'average'`: Average of left and right rotations (most stable, recommended)
  - `'extreme'`: Use the more extreme rotation (ensures full correction, may over-rotate)
  - `'weighted'`: Weight by line quality (length + proximity)

### Canny Edge Detection

#### `canny_low`
- **Default**: `30`
- **Description**: Lower threshold for Canny edge detection. Lower values detect more edges but may include noise.

#### `canny_high`
- **Default**: `100`
- **Description**: Upper threshold for Canny edge detection. Lower values improve edge continuity.

#### `canny_blur_size`
- **Default**: `7`
- **Description**: Gaussian blur kernel size (must be odd). Larger values reduce noise but may blur edges.

#### `canny_blur_sigma`
- **Default**: `1.5`
- **Description**: Gaussian blur sigma. Controls blur strength.

### Morphological Operations

#### `morph_close_kernel`
- **Default**: `3`
- **Description**: Kernel size for morphological closing (connects gaps in edges).

#### `morph_open_kernel`
- **Default**: `2`
- **Description**: Kernel size for morphological opening (removes noise).

### Hough Line Detection

#### `hough_min_line_length_ratio`
- **Default**: `0.5` (50% of height)
- **Description**: Minimum line length as ratio of image height.

#### `hough_min_line_length_absolute`
- **Default**: `100` pixels
- **Description**: Absolute minimum line length in pixels.

#### `hough_threshold_ratio`
- **Default**: `0.15` (15% of height)
- **Description**: Hough threshold as ratio of image height. Higher = fewer but more reliable lines.

#### `hough_threshold_absolute`
- **Default**: `50` pixels
- **Description**: Absolute minimum Hough threshold.

#### `hough_max_line_gap`
- **Default**: `5` pixels
- **Description**: Maximum gap in a line (pixels). Smaller values require more continuous lines.

### Line Filtering

#### `angle_tolerance_degrees`
- **Default**: `15.0` degrees
- **Description**: Maximum deviation from vertical. Lines outside this range are filtered out.

#### `spatial_tolerance_ratio`
- **Default**: `0.4` (40% of width)
- **Description**: Edge position tolerance as ratio of strip width.

#### `use_spatial_filter`
- **Default**: `true`
- **Description**: Whether to filter lines by expected edge position. Disable to detect edges anywhere.

### Line Selection

#### `lines_per_side`
- **Default**: `3`
- **Description**: Number of best lines to consider per side (left/right).

#### `line_score_length_weight`
- **Default**: `0.6`
- **Description**: Weight for line length in scoring (0.0-1.0).

#### `line_score_proximity_weight`
- **Default**: `0.4`
- **Description**: Weight for proximity to expected edge in scoring (0.0-1.0).

### Region Expansion

#### `expand_ratio`
- **Default**: `0.05` (5%)
- **Description**: Expand YOLO bounding box by this ratio to catch edges slightly outside.

### Fallback Parameters

#### `min_contour_area`
- **Default**: `100` pixels²
- **Description**: Minimum contour area for minAreaRect fallback method.

## Usage Examples

### Example 1: More Aggressive Edge Detection

For images with weak edges, lower Canny thresholds and increase angle tolerance:

```python
config = {
    'orientation': {
        'canny_low': 20,  # Lower threshold
        'canny_high': 80,  # Lower threshold
        'angle_tolerance_degrees': 20.0,  # Allow more deviation
        'hough_threshold_ratio': 0.1,  # Lower threshold (more lines)
    }
}
```

### Example 2: Stricter Detection (Fewer False Positives)

For images with many edge-like features, use stricter parameters:

```python
config = {
    'orientation': {
        'canny_low': 40,
        'canny_high': 120,
        'angle_tolerance_degrees': 10.0,  # Stricter angle filter
        'hough_threshold_ratio': 0.2,  # Higher threshold (fewer lines)
        'hough_min_line_length_ratio': 0.6,  # Longer lines only
        'use_spatial_filter': True,  # Only near expected edges
    }
}
```

### Example 3: Average Rotation (Recommended)

Use average rotation for most stable results:

```python
config = {
    'orientation': {
        'rotation_combination': 'average',  # Most stable
        'lines_per_side': 3,  # Use top 3 lines per side
    }
}
```

### Example 4: Extreme Rotation (Full Correction)

If under-rotation is a problem, use extreme rotation:

```python
config = {
    'orientation': {
        'rotation_combination': 'extreme',  # Use more extreme angle
        'max_rotation_angle': 30.0,  # Allow larger corrections
    }
}
```

## Debug Visualization

When debug mode is enabled, the visualization shows:

- **Green rectangle**: Original YOLO bounding box
- **Magenta rectangle**: Expanded region (5% larger)
- **Gray lines**: All detected lines (after filtering)
- **Yellow lines**: Selected lines (top N from each side)
- **Text annotations**: Detection parameters and results

## Troubleshooting

### Problem: Under-rotation (not rotating enough)

**Solutions**:
1. Increase `angle_tolerance_degrees` to allow more deviation
2. Use `rotation_combination: 'extreme'` instead of `'average'`
3. Lower `canny_low` and `canny_high` to detect more edges
4. Lower `hough_threshold_ratio` to detect more lines
5. Increase `lines_per_side` to consider more lines

### Problem: Over-rotation (rotating too much)

**Solutions**:
1. Use `rotation_combination: 'average'` instead of `'extreme'`
2. Decrease `max_rotation_angle` to limit maximum rotation
3. Increase `hough_threshold_ratio` to get fewer, more reliable lines
4. Increase `hough_min_line_length_ratio` to require longer lines

### Problem: No lines detected

**Solutions**:
1. Lower `canny_low` and `canny_high` thresholds
2. Lower `hough_threshold_ratio` and `hough_threshold_absolute`
3. Decrease `hough_min_line_length_ratio` and `hough_min_line_length_absolute`
4. Increase `angle_tolerance_degrees` to allow more deviation
5. Set `use_spatial_filter: false` to detect edges anywhere
6. Increase `expand_ratio` to search in larger area

### Problem: Detecting wrong edges (inner edges instead of outer)

**Solutions**:
1. Enable `use_spatial_filter: true` to focus on expected edge positions
2. Decrease `spatial_tolerance_ratio` to be more strict about position
3. Increase `line_score_proximity_weight` to favor lines near expected edges
4. Increase `hough_min_line_length_ratio` to prefer longer lines (usually outer edges)

## Environment Variables

All parameters can be set via environment variables:

```bash
export REFINEMENT_CANNY_LOW=30
export REFINEMENT_CANNY_HIGH=100
export REFINEMENT_ROTATION_COMBINATION=average
export REFINEMENT_ANGLE_TOLERANCE=15.0
# ... etc
```

## Testing

Use the test script to experiment with different settings:

```bash
python scripts/test_refinement.py tests/fixtures/your_image.jpg
```

Then review the results in the web interface:

```bash
# Start Flask app (includes review interface)
python app.py

# Visit http://localhost:5000/review
```

## Next Steps

1. Test with your images using default settings
2. Review debug visualizations to see what's being detected
3. Adjust parameters based on your specific images
4. Compare results using the web review interface

