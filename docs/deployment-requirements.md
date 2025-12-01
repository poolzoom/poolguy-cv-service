# PoolGuy CV Service - Deployment Requirements

This document outlines all system requirements, dependencies, and configuration needed to deploy the PoolGuy CV Service in production.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Dependencies](#python-dependencies)
3. [System Dependencies](#system-dependencies)
4. [Model Files](#model-files)
5. [Environment Variables](#environment-variables)
6. [Deployment Platforms](#deployment-platforms)
7. [Installation Steps](#installation-steps)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows Server
- **Python**: 3.9 or higher
- **RAM**: 2GB minimum (4GB+ recommended for production)
- **Disk Space**: 1GB minimum (for dependencies and models)
- **CPU**: 2 cores minimum (4+ cores recommended)

### Recommended for Production

- **RAM**: 8GB+
- **CPU**: 4+ cores
- **Disk Space**: 5GB+ (for models, logs, and temporary files)
- **Network**: Stable internet connection (for OpenAI API calls)

## Python Dependencies

All Python dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **Flask 3.0.0**: Web framework
- **Flask-CORS 4.0.0**: Cross-origin resource sharing
- **opencv-python 4.8.1.78**: Computer vision library
- **numpy 1.24.3**: Numerical operations
- **colormath 3.0.0**: Color science (CIEDE2000)
- **Pillow 10.1.0**: Image processing
- **requests 2.31.0**: HTTP client
- **python-dotenv 1.0.0**: Environment variable management
- **scikit-learn 1.6.1**: Machine learning (PCA rotation detection)
- **pytesseract 0.3.10**: OCR fallback for text extraction

## System Dependencies

### Required System Packages

#### Ubuntu/Debian

```bash
# Tesseract OCR (required for text extraction fallback)
sudo apt-get update
sudo apt-get install -y tesseract-ocr

# Additional language packs (optional, for non-English text)
sudo apt-get install -y tesseract-ocr-eng tesseract-ocr-osd
```

#### macOS

```bash
# Tesseract OCR
brew install tesseract

# Optional: Additional language packs
brew install tesseract-lang
```

#### CentOS/RHEL

```bash
# Tesseract OCR
sudo yum install -y tesseract tesseract-langpack-eng
```

### Verification

After installation, verify Tesseract is available:

```bash
tesseract --version
# Should output: tesseract 5.x.x
```

## Model Files

The service requires YOLO model files for object detection. These must be present in the `models/` directory:

### Required Model Files

```
models/
├── best.pt                          # Main strip detection model
├── pad_detection/
│   └── best.pt                      # Pad detection model
└── yolo_strip_detection/
    └── weights/
        └── best.pt                  # Alternative strip detection model
```

### Model Path Configuration

Models are configured via environment variable `YOLO_MODEL_PATH` (defaults to `./models/best.pt`).

**Important**: Ensure model files are included in deployment or downloaded separately. Model files are typically large (50-200MB each) and should be stored separately from code in production.

## Environment Variables

Create a `.env` file in the project root with the following variables:

### Required Variables

```bash
# OpenAI API Key (required for bottle pipeline text extraction)
OPENAI_API_KEY=sk-...

# Flask Configuration
PORT=5000
FLASK_DEBUG=False
```

### Optional Variables

```bash
# Upload directory for temporary files
UPLOAD_FOLDER=/tmp/poolguy-uploads

# Experiments directory for debug outputs
EXPERIMENTS_DIR=experiments

# YOLO Model Configuration
YOLO_MODEL_PATH=./models/best.pt
YOLO_IMG_SIZE=640
YOLO_CONFIDENCE_THRESHOLD=0.01

# Logging
LOG_LEVEL=INFO
```

### Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for Vision API calls |
| `PORT` | No | `5000` | Port for Flask server |
| `FLASK_DEBUG` | No | `False` | Enable Flask debug mode |
| `UPLOAD_FOLDER` | No | `/tmp/poolguy-uploads` | Directory for temporary uploads |
| `EXPERIMENTS_DIR` | No | `experiments` | Directory for debug outputs |
| `YOLO_MODEL_PATH` | No | `./models/best.pt` | Path to YOLO model file |
| `YOLO_IMG_SIZE` | No | `640` | Image size for YOLO inference |
| `YOLO_CONFIDENCE_THRESHOLD` | No | `0.01` | Confidence threshold for detections |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Deployment Platforms

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model files (or mount as volume)
COPY models/ ./models/

# Set environment variables
ENV PORT=5000
ENV FLASK_DEBUG=False

# Expose port
EXPOSE 5000

# Run with Gunicorn (install gunicorn in requirements.txt for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  poolguy-cv:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=5000
      - FLASK_DEBUG=False
    volumes:
      - ./models:/app/models
      - ./experiments:/app/experiments
    restart: unless-stopped
```

### AWS EC2 / Linux Server

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.9 python3-pip tesseract-ocr
   ```

2. **Create virtual environment**:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install gunicorn  # For production WSGI server
   ```

4. **Set up systemd service** (optional):
   ```ini
   # /etc/systemd/system/poolguy-cv.service
   [Unit]
   Description=PoolGuy CV Service
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/opt/poolguy-cv-service
   Environment="PATH=/opt/poolguy-cv-service/venv/bin"
   Environment="OPENAI_API_KEY=sk-..."
   ExecStart=/opt/poolguy-cv-service/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

5. **Start service**:
   ```bash
   sudo systemctl enable poolguy-cv
   sudo systemctl start poolguy-cv
   ```

### Heroku

1. **Add buildpacks**:
   ```bash
   heroku buildpacks:add heroku/python
   heroku buildpacks:add https://github.com/heroku/heroku-buildpack-tesseract
   ```

2. **Set environment variables**:
   ```bash
   heroku config:set OPENAI_API_KEY=sk-...
   heroku config:set PORT=5000
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Kubernetes

See Kubernetes-specific deployment documentation for:
- ConfigMaps for environment variables
- Secrets for API keys
- Persistent volumes for model files
- Deployment and Service manifests

## Installation Steps

### Step-by-Step Installation

1. **Clone or copy project files**:
   ```bash
   git clone <repository-url>
   cd poolguy-cv-service
   ```

2. **Create virtual environment**:
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install system dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y tesseract-ocr

   # macOS
   brew install tesseract
   ```

4. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Install production WSGI server** (optional, for production):
   ```bash
   pip install gunicorn
   ```

6. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Verify model files exist**:
   ```bash
   ls -lh models/best.pt
   # Ensure model files are present
   ```

8. **Test installation**:
   ```bash
   python -c "import cv2; import pytesseract; print('All dependencies OK')"
   tesseract --version
   ```

## Verification

### Health Check

After deployment, verify the service is running:

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "PoolGuy CV Service",
  "version": "1.0.0"
}
```

### Test Endpoints

1. **Test strip extraction**:
   ```bash
   curl -X POST http://localhost:5000/extract-colors \
     -H "Content-Type: application/json" \
     -d '{"image_path": "/path/to/test-strip.jpg", "expected_pad_count": 6}'
   ```

2. **Test bottle pipeline**:
   ```bash
   curl -X POST http://localhost:5000/map-bottle-pads \
     -H "Content-Type: application/json" \
     -d '{"image_paths": ["/path/to/bottle.jpg"]}'
   ```

### Dependency Verification

Run these commands to verify all dependencies:

```bash
# Python version
python --version  # Should be 3.9+

# Python packages
pip list | grep -E "Flask|opencv|numpy|pytesseract"

# Tesseract
tesseract --version

# OpenAI API key (if set)
echo $OPENAI_API_KEY | cut -c1-10  # Should show first 10 chars
```

## Troubleshooting

### Common Issues

#### 1. Tesseract Not Found

**Error**: `pytesseract.pytesseract.TesseractNotFoundError`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr

# macOS
brew install tesseract

# Verify installation
tesseract --version
```

#### 2. OpenAI API Key Missing

**Error**: `OpenAI API key not provided`

**Solution**:
```bash
# Set environment variable
export OPENAI_API_KEY=sk-...

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

#### 3. Model Files Not Found

**Error**: `FileNotFoundError: models/best.pt`

**Solution**:
- Ensure model files are present in `models/` directory
- Or set `YOLO_MODEL_PATH` environment variable to correct path
- Model files should be included in deployment or mounted as volumes

#### 4. Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Change port in .env
PORT=5001

# Or kill process using port
lsof -ti:5000 | xargs kill
```

#### 5. OpenCV Import Error

**Error**: `ImportError: No module named cv2`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall opencv-python
pip install --upgrade opencv-python
```

#### 6. Memory Issues

**Symptoms**: Service crashes or becomes unresponsive

**Solution**:
- Increase server RAM
- Reduce number of Gunicorn workers: `gunicorn -w 2 ...`
- Process smaller images or add image resizing
- Monitor memory usage: `htop` or `free -h`

#### 7. Timeout Issues

**Symptoms**: OpenAI API calls timeout

**Solution**:
- Increase Gunicorn timeout: `gunicorn --timeout 120 ...`
- Check network connectivity
- Implement retry logic (already in code)
- Consider using Tesseract fallback for text extraction

### Logs

Check logs for detailed error information:

```bash
# If using systemd
sudo journalctl -u poolguy-cv -f

# If using Gunicorn directly
# Logs will be in stdout/stderr or configured log file

# Application logs
tail -f experiments/debug.log  # If debug logging enabled
```

## Production Checklist

Before deploying to production, ensure:

- [ ] All system dependencies installed (Tesseract OCR)
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Model files present in `models/` directory
- [ ] Environment variables configured (`.env` file or system environment)
- [ ] `OPENAI_API_KEY` set and valid
- [ ] `FLASK_DEBUG=False` in production
- [ ] Production WSGI server installed (Gunicorn)
- [ ] Health check endpoint responding
- [ ] Firewall rules configured (port 5000 or configured port)
- [ ] Logging configured and monitored
- [ ] Backup strategy for model files
- [ ] Monitoring and alerting set up
- [ ] SSL/TLS configured (if exposed to internet)
- [ ] Rate limiting configured (if needed)

## Additional Resources

- [Flask Deployment Guide](https://flask.palletsprojects.com/en/3.0.x/deploying/)
- [Gunicorn Documentation](https://gunicorn.org/)
- [Tesseract OCR Documentation](https://tesseract-ocr.github.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## Support

For deployment issues or questions, refer to:
- Project README: `README.md`
- Development Guide: `docs/development-guide.md`
- API Reference: `docs/api-reference.md`
- Architecture Documentation: `docs/architecture.md`







