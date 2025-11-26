"""
PoolGuy CV Service - Flask Application
Computer Vision service for test strip color extraction and analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import logging
import time
from dotenv import load_dotenv
import os
from typing import Dict, Optional, Tuple

# Import services
from services.pipeline import PipelineService
from services.utils.image_quality import ImageQualityService
from utils.image_loader import load_image
from systems.web_review_server import create_review_routes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Laravel integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/tmp/poolguy-uploads')

# Initialize services
pipeline_service = PipelineService()
image_quality_service = ImageQualityService()

# Register review routes
experiments_dir = os.getenv('EXPERIMENTS_DIR', 'experiments')
create_review_routes(app, experiments_dir)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'poolguy-cv-service',
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__
    })


def validate_extract_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate extract colors request.
    
    Args:
        data: Request JSON data
    
    Returns:
        Tuple of (is_valid, error_message, error_code)
    """
    if not isinstance(data, dict):
        return False, 'Request must be JSON object', 'INVALID_PARAMETER'
    
    if 'image_path' not in data:
        return False, 'image_path is required', 'MISSING_PARAMETER'
    
    if not isinstance(data['image_path'], str) or not data['image_path'].strip():
        return False, 'image_path must be a non-empty string', 'INVALID_PARAMETER'
    
    if 'expected_pad_count' in data:
        pad_count = data['expected_pad_count']
        if not isinstance(pad_count, int) or not (4 <= pad_count <= 7):
            return False, 'expected_pad_count must be an integer between 4 and 7', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/extract-colors', methods=['POST'])
def extract_colors():
    """
    Extract pad colors from test strip image.
    
    Pipeline: Image → Strip Detection → Pad Detection → Color Extraction
    
    Request (JSON):
    - image_path: Path to test strip image (S3 URL or local path)
    - expected_pad_count: Number of pads expected (4-7, default: 6)
    - normalize_white: Whether to apply white balance normalization (default: true)
    
    Returns:
    - Array of LAB color vectors for each pad
    - Confidence scores
    - Pad coordinates (absolute)
    - Processing time
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'MISSING_PARAMETER'
            }), 400
        
        # Validate request
        is_valid, error_msg, error_code = validate_extract_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        image_path = data.get('image_path')
        expected_pad_count = data.get('expected_pad_count', 6)
        normalize_white = data.get('normalize_white', True)
        
        logger.info(f'Processing image: {image_path}, expected pads: {expected_pad_count}')
        
        # Load image
        try:
            image = load_image(image_path)
        except Exception as e:
            logger.error(f'Failed to load image: {e}', exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Failed to load image: {str(e)}',
                'error_code': 'IMAGE_LOAD_ERROR'
            }), 400
        
        # Process through pipeline
        result = pipeline_service.process_image(
            image=image,
            image_name=os.path.basename(image_path) if image_path else 'unknown',
            expected_pad_count=expected_pad_count,
            normalize_white=normalize_white
        )
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Processing failed'),
                'error_code': result.get('error_code', 'PROCESSING_ERROR')
            }), 500
        
        # Format response
        processing_time_ms = int((time.time() - start_time) * 1000)
        response_data = result.get('data', {})
        response_data['processing_time_ms'] = processing_time_ms
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        logger.error(f'Error in extract_colors: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


def validate_quality_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate image quality request.
    
    Args:
        data: Request JSON data
    
    Returns:
        Tuple of (is_valid, error_message, error_code)
    """
    if not isinstance(data, dict):
        return False, 'Request must be JSON object', 'INVALID_PARAMETER'
    
    if 'image_path' not in data:
        return False, 'image_path is required', 'MISSING_PARAMETER'
    
    if not isinstance(data['image_path'], str) or not data['image_path'].strip():
        return False, 'image_path must be a non-empty string', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/validate-image-quality', methods=['POST'])
def validate_image_quality():
    """
    Validate image quality before processing
    
    Checks:
    - Brightness
    - Contrast
    - Focus (blur detection)
    
    Returns:
    - Quality metrics
    - Pass/fail status
    - Specific error messages if validation fails
    - Recommendations
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'error_code': 'MISSING_PARAMETER'
            }), 400
        
        # Validate request
        is_valid, error_msg, error_code = validate_quality_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        image_path = data.get('image_path')
        
        logger.info(f'Validating image quality: {image_path}')
        
        # Validate quality
        result = image_quality_service.validate_quality(image_path)
        
        if not result.get('success'):
            # Return error response
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'error_code': result.get('error_code', 'IMAGE_QUALITY_ERROR')
            }), 500
        
        # Format response according to API spec
        return jsonify(result)
        
    except Exception as e:
        logger.error(f'Error in validate_image_quality: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500




if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f'Starting PoolGuy CV Service on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)






