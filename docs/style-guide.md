# PoolGuy CV Service - Style Guide

## File Naming Conventions

### General Rules

1. **NO ALL CAPS filenames** - Except for `README.md` which follows standard convention
2. **Use kebab-case** for documentation files (markdown)
3. **Use snake_case** for Python files
4. **Use descriptive names** that clearly indicate the file's purpose

### File Naming Examples

#### ✅ Correct
- `README.md` (standard convention, only exception to ALL CAPS rule)
- `development-guide.md`
- `api-reference.md`
- `strip-detection-critical-review.md`
- `color_extraction.py`
- `test_yolo_detection.py`

#### ❌ Incorrect
- `IMPLEMENTATION.md` → should be `implementation.md`
- `REFINEMENT_SYSTEM.md` → should be `refinement-system.md`
- `IMAGE_ANALYSIS_REPORT.md` → should be `image-analysis-report.md`
- `STRIP_DETECTION_CRITICAL_REVIEW.md` → should be `strip-detection-critical-review.md`

### Directory Structure

```
poolguy-cv-service/
├── app.py                      # Main application
├── README.md                   # Project overview (ONLY ALL CAPS file allowed)
├── docs/                       # Documentation
│   ├── style-guide.md         # This file
│   ├── development-guide.md
│   ├── api-reference.md
│   └── architecture.md
├── services/                   # Business logic
│   ├── color_extraction.py
│   └── strip_detection.py
├── tests/                      # Test files
│   └── test_color_extraction.py
└── scripts/                    # Utility scripts
    └── readme-refinement.md   # Script documentation
```

### Python Code Style

- Follow PEP 8
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Maximum line length: 100 characters (soft), 120 (hard)

### Documentation Style

- Use `kebab-case` for markdown filenames
- Use clear, descriptive titles
- Include table of contents for long documents
- Use proper markdown formatting

## Enforcement

All new files must follow these conventions. Existing files should be renamed to match these standards.


