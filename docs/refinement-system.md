# Strip Refinement System

## Overview

The strip refinement system provides an iterative development workflow for refining YOLO-detected strip regions. It implements a 5-step refinement pipeline and provides tools for experimentation and parameter tuning.

## System Components

### Core Services

- **`services/refinement/strip_refiner.py`**: Main orchestrator that coordinates all refinement steps
- **`services/refinement/methods/`**: Individual refinement method implementations
  - `orientation_normalizer.py`: Rotates strip to vertical (PCA or Hough)
  - `intensity_projector.py`: Projects intensities to find dense regions
  - `edge_detector.py`: Detects strip edges (gradient, white backing, variance)
  - `bbox_tightener.py`: Tightens bounding box by clamping inward
  - `padding_applier.py`: Applies controlled padding for warping

### Configuration

- **`config/refinement_config.py`**: Centralized parameter configuration
  - Base configuration with environment variable overrides
  - Presets: `conservative`, `aggressive`, `balanced`
  - Easy parameter adjustment

### Experiment Tools

- **`scripts/refinement_experiments.py`**: Run experiments with different parameters
- **`scripts/review_refinement.py`**: Review and compare experiment results
- **`scripts/test_refinement.py`**: Quick test on single image

### Evaluation

- **`services/refinement/evaluator.py`**: Calculate quality metrics
  - Tightness score
  - Background removal score
  - Overall quality score

## Quick Start

### 1. Test on Single Image

```bash
python scripts/test_refinement.py tests/fixtures/PXL_20250427_161114135.jpg
```

### 2. Run Experiment

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/*.jpg \
  --preset balanced \
  --output experiments/refinement/baseline
```

### 3. Review Results

```bash
python scripts/review_refinement.py \
  --experiment experiments/refinement/baseline
```

### 4. Adjust and Retry

```bash
python scripts/refinement_experiments.py \
  --images tests/fixtures/*.jpg \
  --preset balanced \
  --override padding.pixels=8 \
  --output experiments/refinement/improved
```

## Refinement Pipeline

The system performs these steps in order:

1. **Orientation Normalization** (A)
   - Rotates strip to vertical using PCA or Hough line detection
   - Config: `orientation.method`, `orientation.pca_threshold`

2. **Intensity Projection** (B)
   - Projects intensities vertically to find dense/consistent regions
   - Config: `projection.window_size`, `projection.smoothing`, `projection.sigma`

3. **Edge Detection** (C)
   - Identifies strip edges using multiple methods:
     - Gradient magnitude along left/right
     - White backing detection (consistent white border)
     - Color variance thresholding
   - Config: `edge_detection.*`

4. **Bounding Box Tightening** (D)
   - Clamps inward on all sides until:
     - Edges stabilize
     - Background variance exceeds threshold
     - White background disappears
   - Config: `tightening.*`

5. **Padding Application** (E)
   - Applies controlled padding (6-12px) after tightening
   - Ensures warping sees the whole strip without cutting pads
   - Config: `padding.*`

## Iterative Workflow

1. **Run baseline experiment** with default settings
2. **Review results** - check quality scores and visualizations
3. **Identify issues** - which step is failing?
4. **Adjust parameters** - edit config or use `--override`
5. **Run new experiment** - test with adjusted parameters
6. **Compare results** - side-by-side comparison
7. **Repeat** until satisfied

## Integration

Once parameters are tuned, integrate into main pipeline:

```python
from services.refinement import StripRefiner
from config.refinement_config import get_config

# Initialize refiner with tuned config
refiner = StripRefiner(get_config('balanced'))  # or custom config

# Refine YOLO detection
refined_strip = refiner.refine(image, yolo_bbox, debug)
```

## Dependencies

- **scikit-learn**: Optional, for PCA-based orientation (falls back to Hough if not available)
- All other dependencies already in requirements.txt

## File Structure

```
services/refinement/
  ├── __init__.py
  ├── base_refiner.py
  ├── strip_refiner.py
  ├── evaluator.py
  └── methods/
      ├── __init__.py
      ├── orientation_normalizer.py
      ├── intensity_projector.py
      ├── edge_detector.py
      ├── bbox_tightener.py
      └── padding_applier.py

scripts/
  ├── refinement_experiments.py
  ├── review_refinement.py
  ├── test_refinement.py
  └── readme-refinement.md

config/
  └── refinement_config.py

experiments/
  └── refinement/
      └── {experiment_name}/
          ├── config.json
          ├── results.json
          └── images/
```

## Next Steps

1. Run initial experiments to establish baseline
2. Review results and identify improvement areas
3. Iteratively tune parameters
4. Integrate into `services/strip_detection.py`
5. Validate on full test set

