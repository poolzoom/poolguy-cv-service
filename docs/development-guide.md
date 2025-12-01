# PoolGuy CV Service - Development Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [OpenCV Best Practices](#opencv-best-practices)
6. [Flask API Design](#flask-api-design)
7. [Testing](#testing)
8. [Debugging](#debugging)
9. [Deployment](#deployment)

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment support

### Initial Setup

1. **Clone/Navigate to project directory**
   ```bash
   cd /Users/poolzoom/Documents/Sites/poolguy-cv-service
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Run the service**
   ```bash
   ./run.sh
   # or
   python app.py
   ```

6. **Verify it's working**
   ```bash
   curl http://localhost:5000/health
   ```

## Project Structure

```
poolguy-cv-service/
├── app.py                      # Main Flask application
├── services/                   # Business logic services
│   ├── __init__.py
│   ├── color_extraction.py     # Test strip pad color extraction
│   ├── image_quality.py        # Image quality validation
│   └── color_matching.py       # Color matching algorithms
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── image_loader.py          # Image loading utilities
│   └── color_conversion.py     # Color space conversions
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_color_extraction.py
│   └── test_image_quality.py
├── docs/                       # Documentation
│   └── development-guide.md
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .cursorrules               # Cursor IDE rules
├── .gitignore                 # Git ignore rules
├── README.md                  # Project overview
└── run.sh                     # Run script
```

## Development Workflow

### Daily Development

1. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

2. **Start development server**
   ```bash
   python app.py
   # Flask will auto-reload on code changes
   ```

3. **Make changes**
   - Edit code in your IDE
   - Flask will automatically reload
   - Test endpoints with curl or Postman

4. **Run tests** (when implemented)
   ```bash
   pytest tests/
   ```

5. **Check code style**
   ```bash
   # Install black if needed
   pip install black
   black --check .
   ```

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature**
   - Add service class in `services/`
   - Add route handler in `app.py`
   - Write tests in `tests/`

3. **Test locally**
   - Run service
   - Test endpoints
   - Run unit tests

4. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add color extraction service"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style Guide

Follow **PEP 8** style guide with these specifics:

- **Line length**: 100 characters (soft), 120 (hard)
- **Indentation**: 4 spaces (no tabs)
- **Naming**:
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`

### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Any

def extract_colors(
    image_path: str, 
    pad_count: int
) -> Dict[str, Any]:
    """Extract colors from test strip."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def process_image(image: np.ndarray, options: Dict) -> List[Dict]:
    """
    Process test strip image and extract pad colors.
    
    Args:
        image: OpenCV image array in BGR format
        options: Processing options dictionary
    
    Returns:
        List of pad color dictionaries with LAB values
    
    Raises:
        ValueError: If image is invalid
        RuntimeError: If processing fails
    """
    pass
```

### Error Handling

Always handle errors gracefully:

```python
try:
    result = process_image(image)
    return {'success': True, 'data': result}
except ValueError as e:
    logger.error(f'Invalid input: {e}')
    return {'success': False, 'error': str(e)}
except Exception as e:
    logger.error(f'Unexpected error: {e}', exc_info=True)
    return {'success': False, 'error': 'Internal error'}
```

### Logging

Use Python's logging module:

```python
import logging

logger = logging.getLogger(__name__)

logger.debug('Detailed debug information')
logger.info('Processing image: %s', image_path)
logger.warning('Low confidence score: %f', confidence)
logger.error('Failed to process: %s', error, exc_info=True)
```

## OpenCV Best Practices

### Image Loading

```python
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    """Load image from path or URL."""
    if image_path.startswith('http'):
        # Download from URL (S3 signed URL)
        import requests
        response = requests.get(image_path, timeout=30)
        image = cv2.imdecode(
            np.frombuffer(response.content, np.uint8),
            cv2.IMREAD_COLOR
        )
    else:
        # Load from local path
        image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f'Failed to load image: {image_path}')
    
    return image
```

### Color Space Conversion

```python
# OpenCV uses BGR by default
bgr_image = cv2.imread('image.jpg')

# Convert to RGB for display/processing
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

# Convert to LAB for color matching (perceptually uniform)
lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

# Extract LAB values
L, a, b = cv2.split(lab_image)
```

### Image Processing

```python
# Resize if needed (maintain aspect ratio)
def resize_image(image: np.ndarray, max_width: int = 1920) -> np.ndarray:
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image

# Preprocessing
def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale for some operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Edge detection
    edges = cv2.Canny(denoised, 50, 150)
    
    return edges
```

### Memory Management

```python
# Release images when done
def process_and_cleanup(image_path: str):
    image = load_image(image_path)
    result = process_image(image)
    del image  # Explicit cleanup
    return result
```

## Flask API Design

### Route Structure

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/extract-colors', methods=['POST'])
def extract_colors():
    """Extract pad colors from test strip."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'image_path' not in data:
            return jsonify({
                'success': False,
                'error': 'image_path is required'
            }), 400
        
        # Process
        result = process_extraction(data)
        
        # Return success
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f'Error: {e}', exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### Request Validation

```python
def validate_extract_request(data: Dict) -> tuple[bool, Optional[str]]:
    """Validate extract colors request."""
    if not isinstance(data, dict):
        return False, 'Request must be JSON object'
    
    if 'image_path' not in data:
        return False, 'image_path is required'
    
    if not isinstance(data['image_path'], str):
        return False, 'image_path must be a string'
    
    if 'expected_pad_count' in data:
        pad_count = data['expected_pad_count']
        if not isinstance(pad_count, int) or not (4 <= pad_count <= 7):
            return False, 'expected_pad_count must be between 4 and 7'
    
    return True, None
```

### Response Format

**Success Response:**
```json
{
  "success": true,
  "data": {
    "pads": [
      {
        "pad_index": 0,
        "lab": {"L": 50.0, "a": 0.0, "b": 0.0},
        "confidence": 0.95
      }
    ],
    "overall_confidence": 0.92
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message",
  "error_code": "INVALID_IMAGE"
}
```

## Testing

### Unit Testing with pytest

```python
import pytest
from services.color_extraction import ColorExtractionService

class TestColorExtraction:
    def setup_method(self):
        """Set up test fixtures."""
        self.service = ColorExtractionService()
    
    def test_extract_colors_success(self):
        """Test successful color extraction."""
        image_path = 'tests/fixtures/test-strip.jpg'
        result = self.service.extract_colors(image_path, pad_count=6)
        
        assert result['success'] is True
        assert 'pads' in result['data']
        assert len(result['data']['pads']) == 6
    
    def test_extract_colors_invalid_image(self):
        """Test handling of invalid image."""
        result = self.service.extract_colors('invalid.jpg', pad_count=6)
        
        assert result['success'] is False
        assert 'error' in result
```

### Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=services --cov-report=html

# Run specific test
pytest tests/test_color_extraction.py::TestColorExtraction::test_extract_colors_success
```

## Debugging

### Using Logging

```python
import logging

logger = logging.getLogger(__name__)

# Debug level (detailed)
logger.debug(f'Image dimensions: {image.shape}')

# Info level (important events)
logger.info(f'Processing image: {image_path}')

# Warning level (potential issues)
logger.warning(f'Low confidence: {confidence}')

# Error level (errors)
logger.error(f'Failed to process: {error}', exc_info=True)
```

### Flask Debug Mode

Set in `.env`:
```
FLASK_DEBUG=True
```

This enables:
- Auto-reload on code changes
- Detailed error pages
- Debugger in browser

### Using Python Debugger

```python
import pdb

def process_image(image):
    # Set breakpoint
    pdb.set_trace()
    
    # Your code here
    result = extract_colors(image)
    return result
```

## Deployment

### Production Server

Use Gunicorn for production:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

Set in production:
```
FLASK_DEBUG=False
PORT=5000
LOG_LEVEL=INFO
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Common Issues

### OpenCV Import Error
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall opencv-python
pip install --upgrade opencv-python
```

### Port Already in Use
```bash
# Change PORT in .env
PORT=5001

# Or kill process using port
lsof -ti:5000 | xargs kill
```

### Memory Issues
- Process smaller images
- Release images after processing
- Use image resizing for large images

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)












