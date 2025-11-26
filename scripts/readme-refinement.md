# Strip Refinement System - Usage Guide

## Overview

The strip refinement system allows you to iteratively develop and tune refinement parameters for YOLO-detected strips. It provides a complete workflow: try → review → adjust → retry.

## Quick Start

### 1. Run Your First Experiment

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/PXL_20250427_161114135.jpg \
  --preset balanced \
  --output experiments/refinement/first_test
```

### 2. Review Results

```bash
python scripts/review_refinement.py \
  --experiment experiments/refinement/first_test
```

### 3. Adjust Parameters and Retry

Edit `config/refinement_config.py` or use command-line overrides:

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/PXL_20250427_161114135.jpg \
  --preset balanced \
  --override padding.pixels=8 \
  --override tightening.edge_stability_threshold=0.03 \
  --output experiments/refinement/test_v2
```

### 4. Compare Experiments

```bash
python scripts/review_refinement.py \
  --experiment experiments/refinement/test_v1 \
  --compare experiments/refinement/test_v2
```

## Configuration

### Using Presets

- `--preset conservative`: More padding, less aggressive tightening
- `--preset aggressive`: Less padding, more aggressive tightening
- `--preset balanced`: Default balanced settings

### Custom Configuration

Create a JSON config file:

```json
{
  "orientation": {
    "method": "pca",
    "pca_threshold": 0.1
  },
  "padding": {
    "pixels": 10,
    "adaptive": true,
    "min_padding": 6,
    "max_padding": 12
  },
  "tightening": {
    "edge_stability_threshold": 0.05,
    "background_variance_threshold": 20.0,
    "white_background_threshold": 240
  }
}
```

Use it with:

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/*.jpg \
  --config my_config.json \
  --output experiments/refinement/custom_test
```

### Command-Line Overrides

Override specific parameters:

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/*.jpg \
  --override padding.pixels=8 \
  --override tightening.edge_stability_threshold=0.03 \
  --override edge_detection.gradient_threshold=0.25
```

## Understanding Results

### Metrics

- **Quality Score** (0-1): Overall refinement quality
  - Higher = better fit, less background
- **Tightness Score** (0-1): How well the bbox fits the strip
  - Higher = tighter fit
- **Background Removal Score** (0-1): How much background was removed
  - Higher = more background removed
- **Size Reduction** (0-1): Percentage of original area removed
  - Higher = more area removed

### Visualizations

Each experiment creates:
- `images/{image_name}_comparison.jpg`: Before/after comparison
- `images/{image_name}/`: Step-by-step debug images
- `results.json`: All metrics and metadata

## Refinement Steps

The system performs these steps in order:

1. **Orientation Normalization** (A)
   - Rotates strip to vertical using PCA or Hough lines
   - Config: `orientation.method`, `orientation.pca_threshold`

2. **Intensity Projection** (B)
   - Projects intensities vertically to find dense regions
   - Config: `projection.window_size`, `projection.smoothing`, `projection.sigma`

3. **Edge Detection** (C)
   - Identifies strip edges using:
     - Gradient magnitude
     - White backing detection
     - Color variance
   - Config: `edge_detection.*`

4. **Bounding Box Tightening** (D)
   - Clamps inward until edges stabilize or background detected
   - Config: `tightening.*`

5. **Padding Application** (E)
   - Applies controlled padding for warping
   - Config: `padding.*`

## Iterative Workflow

1. **Start with default preset:**
   ```bash
   python scripts/refinement_experiments.py \
     --images tests/fixtures/*.jpg \
     --preset balanced \
     --output experiments/refinement/baseline
   ```

2. **Review and identify issues:**
   ```bash
   python scripts/review_refinement.py --experiment experiments/refinement/baseline
   ```
   - Check quality scores
   - Look at visualizations
   - Identify which step needs adjustment

3. **Adjust parameters:**
   - Edit `config/refinement_config.py`
   - Or use `--override` flags
   - Focus on the step that's failing

4. **Run new experiment:**
   ```bash
   python scripts/refinement_experiments.py \
     --images tests/fixtures/*.jpg \
     --preset balanced \
     --override [your changes] \
     --output experiments/refinement/improved_v1
   ```

5. **Compare results:**
   ```bash
   python scripts/review_refinement.py \
     --experiment experiments/refinement/baseline \
     --compare experiments/refinement/improved_v1
   ```

6. **Repeat until satisfied**

## Tips

- **Start conservative**: Use `--preset conservative` first
- **One change at a time**: Adjust one parameter set per experiment
- **Test on multiple images**: Use `--images tests/fixtures/*.jpg`
- **Check visualizations**: Look at step-by-step images in `images/` directory
- **Focus on failing cases**: Identify which images fail and why

## Troubleshooting

### No improvements in metrics
- Check if refinement is actually running (look for debug images)
- Verify YOLO detection is working first
- Try more aggressive tightening parameters

### Refinement fails
- Check logs for error messages
- Verify image is valid and YOLO detected a strip
- Try more conservative parameters

### Results look wrong
- Review step-by-step images in `images/{image_name}/`
- Check which step is failing
- Adjust parameters for that specific step

## Integration

Once you're satisfied with parameters, integrate into the main pipeline:

```python
from services.refinement import StripRefiner
from config.refinement_config import get_config

refiner = StripRefiner(get_config('balanced'))
refined_strip = refiner.refine(image, yolo_bbox, debug)
```

