# Hardcoded Values Review

This document lists all hardcoded values found in the refinement codebase that should be moved to configuration.

## Summary

**Total hardcoded values found: 50+**

These values are scattered across multiple files and should be centralized in `config/refinement_config.py` for easy tuning and experimentation.

---

## 1. `services/refinement/methods/intensity_projector.py`

### Threshold Multipliers (Lines 183, 225, 248, 262, 270)
- **`0.3`** - Vertical projection threshold multiplier (`_find_edges`)
- **`0.1`** - Horizontal projection threshold multiplier (`_find_edges_aggressive`)
- **`0.5`** - Top edge gradient threshold multiplier
- **`0.6`** - Bottom edge negative gradient threshold multiplier
- **`0.15`** - Bottom edge fallback threshold multiplier

**Should be in config:**
```python
'projection': {
    'vertical_threshold_multiplier': 0.3,
    'horizontal_threshold_multiplier': 0.1,
    'top_gradient_multiplier': 0.5,
    'bottom_gradient_multiplier': 0.6,
    'bottom_fallback_multiplier': 0.15,
}
```

### Search Ranges (Lines 247, 257, 261, 271)
- **`50`** - Top edge search range (rows to check)
- **`50`** - Bottom edge start search offset (pixels after last high point)
- **`100`** - Bottom edge gradient search range (pixels to search backwards)
- **`150`** - Bottom edge fallback search range (pixels to search backwards)

**Should be in config:**
```python
'projection': {
    'top_search_range': 50,
    'bottom_start_offset': 50,
    'bottom_gradient_search_range': 100,
    'bottom_fallback_search_range': 150,
}
```

---

## 2. `services/refinement/methods/orientation_normalizer.py`

### Aspect Ratio Constants (Lines 287, 291, 293, 294)
- **`7.0`** - Expected aspect ratio for vertical strip
- **`0.7`** - Aspect ratio threshold (70% of expected)
- **`0.1`** - Minimum aspect ratio (to prevent division by zero)
- **`3.0`** - Maximum expansion factor multiplier

**Should be in config:**
```python
'orientation': {
    'expected_aspect_ratio': 7.0,
    'aspect_ratio_threshold': 0.7,
    'min_aspect_ratio': 0.1,
    'max_expansion_factor': 3.0,
}
```

---

## 3. `services/refinement/methods/adaptive_params.py`

### Canny Parameter Constants
- **`0.66`** - Median multiplier for low threshold (line 36)
- **`1.33`** - Median multiplier for high threshold (line 37)
- **`30.0`** - Standard deviation normalization factor (line 40)
- **`0.3`** - Contrast factor multiplier (lines 41, 42)
- **`640`** - Reference image width for size normalization (line 45)
- **`480`** - Reference image height for size normalization (line 45)
- **`1.5`** - Maximum size factor (line 45)
- **`0.2`** - Size factor multiplier (lines 46, 47)
- **`10`** - Minimum Canny low threshold (line 50)
- **`100`** - Maximum Canny low threshold (line 50)
- **`50`** - Minimum Canny high threshold (line 51)
- **`255`** - Maximum Canny high threshold (line 51)

**Should be in config:**
```python
'adaptive_canny': {
    'median_low_multiplier': 0.66,
    'median_high_multiplier': 1.33,
    'std_normalization': 30.0,
    'contrast_factor_multiplier': 0.3,
    'reference_width': 640,
    'reference_height': 480,
    'max_size_factor': 1.5,
    'size_factor_multiplier': 0.2,
    'min_low_threshold': 10,
    'max_low_threshold': 100,
    'min_high_threshold': 50,
    'max_high_threshold': 255,
}
```

### Hough Parameter Constants
- **`0.6`** - Base min line length ratio (line 73)
- **`0.05`** - Base threshold ratio (line 74)
- **`0.05`** - Edge density threshold (low) (line 79)
- **`0.15`** - Edge density threshold (high) (line 81)
- **`0.4`** - Min length ratio for low density (line 80)
- **`0.7`** - Min length ratio for high density (line 82)
- **`0.02`** - Threshold ratio for low density (line 92)
- **`0.08`** - Threshold ratio for high density (line 94)
- **`60`** - Absolute minimum line length (line 86)
- **`20`** - Absolute minimum threshold (line 98)
- **`5`** - Absolute minimum max line gap (line 102)
- **`0.01`** - Max line gap ratio (1% of height) (line 102)

**Should be in config:**
```python
'adaptive_hough': {
    'base_min_length_ratio': 0.6,
    'base_threshold_ratio': 0.05,
    'edge_density_low': 0.05,
    'edge_density_high': 0.15,
    'min_length_ratio_low_density': 0.4,
    'min_length_ratio_high_density': 0.7,
    'threshold_ratio_low_density': 0.02,
    'threshold_ratio_high_density': 0.08,
    'min_line_length_absolute': 60,
    'min_threshold_absolute': 20,
    'min_max_line_gap_absolute': 5,
    'max_line_gap_ratio': 0.01,
}
```

---

## 4. `services/refinement/methods/bbox_tightener.py`

### Gradient Normalization (Line 99, 130, 161, 192)
- **`255`** - Gradient normalization factor (used in multiple places)

**Should be in config:**
```python
'tightening': {
    'gradient_normalization': 255.0,
}
```

### Already Configurable (but defaults are hardcoded)
- `edge_stability_threshold` - Default: `0.05` (already in config)
- `background_variance_threshold` - Default: `20.0` (already in config)
- `white_background_threshold` - Default: `240` (already in config)

---

## 5. `services/refinement/methods/edge_detector.py`

### Gradient Threshold (Line 104)
- **`0.3`** - Gradient magnitude threshold (already in config as `gradient_threshold`)

### White Backing Threshold (Line 135)
- **`200`** - White backing threshold (already in config as `white_backing_threshold`)

### Color Variance Threshold (Line 151)
- **`15.0`** - Color variance threshold (already in config as `color_variance_threshold`)

**Note:** These are already in config, but the code doesn't use `self.config.get()` - it uses hardcoded values.

---

## 6. `services/refinement/strip_refiner.py`

### Rotation Threshold (Line 156)
- **`0.1`** - Minimum rotation angle to apply transformation (degrees)

**Should be in config:**
```python
'orientation': {
    'min_rotation_for_transform': 0.1,
}
```

### Confidence (Line 223)
- **`1.0`** - Refinement confidence (hardcoded, could be calculated)

**Should be calculated or configurable:**
```python
'refinement': {
    'default_confidence': 1.0,
}
```

---

## 7. Visual Debugging Constants

### Color Values (Multiple files)
- **`(255, 255, 0)`** - Yellow for original/input regions
- **`(0, 255, 0)`** - Green for refined/output regions
- **`(255, 0, 0)`** - Blue for YOLO detections
- **`(0, 0, 255)`** - Red (if used)

**Should be in config:**
```python
'debug': {
    'color_original': (255, 255, 0),
    'color_refined': (0, 255, 0),
    'color_yolo': (255, 0, 0),
    'color_error': (0, 0, 255),
}
```

### Font Settings (Multiple files)
- **`cv2.FONT_HERSHEY_SIMPLEX`** - Font type
- **`0.8`** - Font scale (multiple places)
- **`1.0`** - Font scale (multiple places)
- **`2`** - Font thickness (multiple places)

**Should be in config:**
```python
'debug': {
    'font_type': 'FONT_HERSHEY_SIMPLEX',
    'font_scale_small': 0.8,
    'font_scale_large': 1.0,
    'font_thickness': 2,
}
```

---

## Recommended Action Plan

1. **Add all missing parameters to `config/refinement_config.py`**
2. **Update all methods to use `self.config.get()` instead of hardcoded values**
3. **Add environment variable support for all new parameters**
4. **Update documentation to explain each parameter**
5. **Test with different parameter sets to ensure they work**

---

## Priority Order

1. **High Priority** - Parameters that directly affect detection accuracy:
   - Threshold multipliers in `intensity_projector.py`
   - Aspect ratio constants in `orientation_normalizer.py`
   - Adaptive parameter constants in `adaptive_params.py`

2. **Medium Priority** - Parameters that affect edge cases:
   - Search ranges in `intensity_projector.py`
   - Rotation thresholds in `strip_refiner.py`

3. **Low Priority** - Visual/debugging parameters:
   - Color values
   - Font settings

---

## Notes

- Some values (like `255` for max pixel value) are fundamental constants and may not need to be configurable
- Some values (like `0.1` minimum aspect ratio) are safety checks to prevent division by zero
- Consider creating parameter presets for different image types (high contrast, low contrast, rotated, etc.)

