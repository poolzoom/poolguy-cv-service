# PoolGuy CV Service

Computer Vision microservice for test strip color extraction and analysis using OpenCV and Flask.

## Overview

This service provides image processing capabilities for PoolGuy.ai's test strip analysis feature:
- Test strip pad detection and color extraction
- LAB color space conversion
- Image quality validation
- Color matching support

## Requirements

- Python 3.9+
- Virtual environment (venv)
- OpenCV 4.8+
- Flask 3.0+

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update as needed:

```bash
cp .env.example .env
```

### 4. Run the Service

```bash
# Development mode
python app.py

# Or with Flask CLI
flask run --host=0.0.0.0 --port=5000
```

The service will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

Returns service status and version information.

### Extract Colors
```
POST /extract-colors
Content-Type: application/json

{
  "image_path": "/path/to/test-strip.jpg",
  "expected_pad_count": 6
}
```

Extracts pad colors from test strip image.

### Validate Image Quality
```
POST /validate-image-quality
Content-Type: application/json

{
  "image_path": "/path/to/test-strip.jpg"
}
```

Validates image quality (brightness, contrast, focus).

## Development

### Project Structure

```
poolguy-cv-service/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── venv/                # Virtual environment (not in git)
```

### Best Practices for Mac Development

1. **Use Virtual Environment**: Always activate venv before running
   ```bash
   source venv/bin/activate
   ```

2. **OpenCV on Mac**: The `opencv-python` package includes pre-built wheels that work well on Mac (including Apple Silicon). No additional system dependencies needed.

3. **Flask Development**: Use Flask's built-in development server for local development:
   ```bash
   flask run --debug
   ```

4. **Production Deployment**: For production, use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Testing

Test the health endpoint:

```bash
curl http://localhost:5000/health
```

## Integration with Laravel

The Laravel application will call this service via HTTP:

```php
$response = Http::post('http://localhost:5000/extract-colors', [
    'image_path' => $s3ImageUrl,
    'expected_pad_count' => 6
]);
```

## Next Steps

1. Implement test strip detection algorithm
2. Implement pad detection (4-7 pads)
3. Implement color extraction
4. Implement LAB color space conversion
5. Implement white normalization
6. Implement image quality validation
7. Add comprehensive error handling
8. Add unit tests

## Troubleshooting

### OpenCV Import Error
If you get `ImportError: No module named cv2`, make sure:
- Virtual environment is activated
- `opencv-python` is installed: `pip install opencv-python`

### Port Already in Use
If port 5000 is in use, change it in `.env`:
```
PORT=5001
```

## License

Internal use for PoolGuy.ai project.







