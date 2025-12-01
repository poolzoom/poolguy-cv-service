"""
PoolGuy CV Service - Flask Application
Computer Vision service for test strip color extraction and analysis
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import cv2
import numpy as np
import logging
import time
import uuid
from dotenv import load_dotenv
import os
from typing import Dict, Optional, Tuple

# Import services
from services.pipeline import PipelineService
from services.pipeline.bottle_pipeline import BottlePipelineService
from services.utils.image_quality import ImageQualityService
from services.utils.debug import DebugContext
from utils.image_loader import load_image
from utils.color_conversion import extract_color_from_region
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

# Determine if we're in production mode
is_production = os.getenv('FLASK_DEBUG', 'False').lower() != 'true'

# Request ID middleware for tracing
@app.before_request
def generate_request_id():
    """Generate or use existing request ID for tracing."""
    g.request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())
    logger.debug(f'[Request {g.request_id}] {request.method} {request.path}')

@app.after_request
def add_request_id_header(response):
    """Add request ID to response headers."""
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    return response


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitize error messages for production.
    
    In production, returns generic messages to prevent information disclosure.
    In development, returns full error details for debugging.
    
    Args:
        error: Exception object
    
    Returns:
        Sanitized error message string
    """
    if is_production:
        # Return generic message in production
        return "An internal error occurred. Please try again later."
    else:
        # Return full error in development
        return str(error)

# Rate limiting configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour", "20 per minute"],
    storage_uri="memory://",  # In-memory storage (use Redis in production for multi-instance)
    headers_enabled=True
)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/tmp/poolguy-uploads')

# Initialize services
pipeline_service = PipelineService()
bottle_pipeline_service = BottlePipelineService()
image_quality_service = ImageQualityService()

# Register review routes (development/debugging only)
# Only register if ENABLE_REVIEW_ROUTES is set to 'true' or FLASK_DEBUG is 'true'
enable_review = os.getenv('ENABLE_REVIEW_ROUTES', 'false').lower() == 'true'
flask_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
if enable_review or flask_debug:
    experiments_dir = os.getenv('EXPERIMENTS_DIR', 'experiments')
    create_review_routes(app, experiments_dir)
    logger.info("Review routes enabled (development mode)")
else:
    logger.info("Review routes disabled (production mode)")


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
        if not isinstance(pad_count, int) or not (3 <= pad_count <= 7):
            return False, 'expected_pad_count must be an integer between 3 and 7', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/extract-colors', methods=['POST'])
@limiter.limit("10 per minute")  # More restrictive for heavy image processing
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
        
        request_id = getattr(g, 'request_id', 'unknown')
        logger.info(f'[Request {request_id}] Processing image: {image_path}, expected pads: {expected_pad_count}')
        
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
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in extract_colors: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
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
@limiter.limit("20 per minute")  # Lighter operation, higher limit
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
        
        request_id = getattr(g, 'request_id', 'unknown')
        logger.info(f'[Request {request_id}] Validating image quality: {image_path}')
        
        # Validate quality (non-blocking - always returns valid=true with warnings)
        result = image_quality_service.validate_quality(image_path)
        
        if not result.get('success'):
            # Return error response only if validation itself failed (not quality issues)
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'error_code': result.get('error_code', 'IMAGE_QUALITY_ERROR')
            }), 500
        
        # Always return valid=true (non-blocking), but include warnings/errors for information
        result['valid'] = True  # Override to always be valid - quality check is informational only
        
        # Format response according to API spec
        return jsonify(result)
        
    except Exception as e:
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in validate_image_quality: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


def validate_bottle_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate map bottle pads request.
    
    Args:
        data: Request JSON data
    
    Returns:
        Tuple of (is_valid, error_message, error_code)
    """
    if not isinstance(data, dict):
        return False, 'Request must be JSON object', 'INVALID_PARAMETER'
    
    # Check for image_paths (array) or image_path (single)
    has_image_paths = 'image_paths' in data and isinstance(data['image_paths'], list) and len(data['image_paths']) > 0
    has_image_path = 'image_path' in data and isinstance(data['image_path'], str) and data['image_path'].strip()
    
    if not has_image_paths and not has_image_path:
        return False, 'Either image_paths (array) or image_path (string) is required', 'MISSING_PARAMETER'
    
    # Validate image_paths if provided
    if has_image_paths:
        for path in data['image_paths']:
            if not isinstance(path, str) or not path.strip():
                return False, 'All image_paths must be non-empty strings', 'INVALID_PARAMETER'
    
    # Validate options if provided
    if 'options' in data:
        options = data['options']
        if not isinstance(options, dict):
            return False, 'options must be a JSON object', 'INVALID_PARAMETER'
        
        if 'detection_method' in options:
            valid_methods = ['yolo', 'openai', 'opencv', 'auto']
            if options['detection_method'] not in valid_methods:
                return False, f'detection_method must be one of: {", ".join(valid_methods)}', 'INVALID_PARAMETER'
        
        if 'ocr_method' in options:
            valid_methods = ['openai', 'tesseract', 'auto']
            if options['ocr_method'] not in valid_methods:
                return False, f'ocr_method must be one of: {", ".join(valid_methods)}', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/map-bottle-pads', methods=['POST'])
@limiter.limit("5 per minute")  # Heavy operation with OpenAI calls
def map_bottle_pads():
    """
    Map pads on test strip bottle images.
    
    Pipeline: Image(s) → Text Extraction → Pad Detection → Reference Square Detection → Color Mapping
    
    Request (JSON):
    - image_paths: Array of image paths (or image_path for single image)
    - options: Optional processing options:
        - detection_method: "yolo", "openai", "opencv", "auto" (default: "auto")
        - ocr_method: "openai", "tesseract", "auto" (default: "auto")
    
    Returns:
    - Array of pads with names, reference ranges, and color mappings
    - Overall confidence
    - Number of images processed
    - Number of pads detected
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
        is_valid, error_msg, error_code = validate_bottle_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        # Get image paths
        if 'image_paths' in data:
            image_paths = data['image_paths']
        else:
            # Single image path
            image_paths = [data['image_path']]
        
        options = data.get('options', {})
        debug_enabled = data.get('debug', False)
        
        request_id = getattr(g, 'request_id', 'unknown')
        logger.info(f'[Request {request_id}] Processing {len(image_paths)} bottle image(s), debug: {debug_enabled}')
        
        # Setup debug context if enabled
        debug = None
        if debug_enabled:
            experiments_dir = os.getenv('EXPERIMENTS_DIR', 'experiments')
            image_name = os.path.basename(image_paths[0]) if image_paths else 'bottle'
            debug = DebugContext(
                enabled=True,
                output_dir=experiments_dir,
                image_name=image_name,
                comparison_tag='bottle_pipeline'
            )
            # Log input
            if image_paths:
                try:
                    first_image = load_image(image_paths[0])
                    if first_image is not None:
                        debug.add_step('00_input', 'Input Image', first_image, {
                            'image_paths': image_paths,
                            'options': options
                        })
                except Exception as e:
                    logger.warning(f'Could not load image for debug: {e}')
        
        # Process through bottle pipeline
        result = bottle_pipeline_service.process_bottle(
            image_paths=image_paths,
            options=options,
            debug=debug
        )
        
        # Save debug log if enabled
        if debug and debug.enabled:
            try:
                debug.save_log()
                if debug._log_dir:
                    result['debug'] = {
                        'enabled': True,
                        'visual_log_path': str(debug._log_dir)
                    }
            except Exception as e:
                logger.warning(f'Failed to save debug log: {e}')
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Processing failed'),
                'error_code': result.get('error_code', 'PROCESSING_ERROR')
            }), 500
        
        # Format response
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'data': {
                'pads': result.get('pads', []),
                'overall_confidence': result.get('overall_confidence', 0.0),
                'images_processed': result.get('images_processed', 0),
                'pads_detected': result.get('pads_detected', 0),
                'processing_time_ms': processing_time_ms
            }
        })
        
    except Exception as e:
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in map_bottle_pads: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


def validate_extract_at_locations_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate extract colors at locations request.
    
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
    
    if 'locations' not in data:
        return False, 'locations is required', 'MISSING_PARAMETER'
    
    if not isinstance(data['locations'], list):
        return False, 'locations must be an array', 'INVALID_PARAMETER'
    
    if len(data['locations']) == 0:
        return False, 'locations array cannot be empty', 'INVALID_PARAMETER'
    
    # Validate each location
    for i, loc in enumerate(data['locations']):
        if not isinstance(loc, dict):
            return False, f'locations[{i}] must be an object', 'INVALID_PARAMETER'
        
        required_fields = ['x', 'y', 'width', 'height']
        for field in required_fields:
            if field not in loc:
                return False, f'locations[{i}].{field} is required', 'MISSING_PARAMETER'
            if not isinstance(loc[field], (int, float)):
                return False, f'locations[{i}].{field} must be a number', 'INVALID_PARAMETER'
            if loc[field] < 0:
                return False, f'locations[{i}].{field} cannot be negative', 'INVALID_PARAMETER'
        
        if loc['width'] == 0 or loc['height'] == 0:
            return False, f'locations[{i}] width and height must be greater than 0', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/extract-colors-at-locations', methods=['POST'])
@limiter.limit("15 per minute")  # Moderate operation
def extract_colors_at_locations():
    """
    Extract colors from specified regions in an image.
    
    For each location, extracts the average color from the center portion
    of the region to avoid edge effects.
    
    Request (JSON):
    - image_path: URL or path to the image
    - locations: Array of regions, each with:
        - x: X coordinate (top-left)
        - y: Y coordinate (top-left)
        - width: Region width
        - height: Region height
    
    Returns:
    - Array of color data for each location:
        - lab: LAB color values (L: 0-100, a: -128 to +127, b: -128 to +127)
        - rgb: RGB color values (r, g, b: 0-255)
        - hex: Hex color string (e.g., "#A4C639")
        - region: Original region coordinates
    - Processing time in milliseconds
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
        is_valid, error_msg, error_code = validate_extract_at_locations_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        image_path = data['image_path']
        locations = data['locations']
        
        logger.info(f'Extracting colors at {len(locations)} locations from: {image_path}')
        
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
        
        # Extract colors from each location
        colors = []
        for loc in locations:
            try:
                x = int(loc['x'])
                y = int(loc['y'])
                width = int(loc['width'])
                height = int(loc['height'])
                
                color_data = extract_color_from_region(image, x, y, width, height)
                color_data['region'] = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                }
                colors.append(color_data)
                
            except Exception as e:
                logger.warning(f'Failed to extract color at location {loc}: {e}')
                # Include error info for this location
                colors.append({
                    'lab': None,
                    'rgb': None,
                    'hex': None,
                    'region': loc,
                    'error': str(e)
                })
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'data': {
                'colors': colors,
                'processing_time_ms': processing_time_ms
            }
        })
        
    except Exception as e:
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in extract_colors_at_locations: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


# =============================================================================
# BOTTLE LABEL ANALYSIS ENDPOINTS (New Simplified API)
# =============================================================================

def validate_analyze_bottle_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate analyze bottle label request.
    
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
    
    # Validate hints if provided
    if 'hints' in data and data['hints'] is not None:
        hints = data['hints']
        if not isinstance(hints, dict):
            return False, 'hints must be a JSON object', 'INVALID_PARAMETER'
        
        if 'expected_pad_count' in hints:
            if not isinstance(hints['expected_pad_count'], int) or hints['expected_pad_count'] < 1:
                return False, 'hints.expected_pad_count must be a positive integer', 'INVALID_PARAMETER'
    
    return True, None, None


@app.route('/analyze-bottle-label', methods=['POST'])
@limiter.limit("5 per minute")  # Heavy operation with OpenAI calls
def analyze_bottle_label():
    """
    Analyze a bottle label image using OpenAI Vision + OpenCV.
    
    Detects brand, test parameters, and color swatches with their values.
    
    Request (JSON):
    - image_path: URL or path to the bottle image
    - hints: Optional hints to improve detection:
        - expected_pad_count: int
        - known_brand: str
        - known_parameters: List[str]
    
    Returns:
    - detection: Is this a test strip bottle? Confidence score
    - brand: Detected brand name and confidence
    - pads: Array of detected parameters with swatches and colors
    - image_dimensions: Width and height of processed image
    - processing_time_ms: Processing time
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
        is_valid, error_msg, error_code = validate_analyze_bottle_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        image_path = data['image_path']
        hints = data.get('hints')
        debug_enabled = data.get('debug', False)
        
        logger.info(f'Analyzing bottle label: {image_path}')
        
        # Setup debug context if enabled
        debug = None
        if debug_enabled:
            debug = DebugContext(
                enabled=True,
                output_dir=os.getenv('EXPERIMENTS_DIR', 'experiments'),
                image_name=os.path.basename(image_path),
                comparison_tag='analyze_bottle'
            )
        
        # Analyze bottle label
        result = bottle_pipeline_service.analyze_bottle_label(
            image_path=image_path,
            hints=hints,
            debug=debug
        )
        
        # Save debug log if enabled
        if debug and debug.enabled:
            try:
                debug.save_log()
            except Exception as e:
                logger.warning(f'Failed to save debug log: {e}')
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Analysis failed'),
                'error_code': result.get('error_code', 'ANALYSIS_ERROR')
            }), 500
        
        return jsonify(result)
        
    except Exception as e:
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in analyze_bottle_label: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


def validate_detect_swatches_request(data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate detect color swatches request.
    
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
    
    # Validate options if provided
    if 'options' in data and data['options'] is not None:
        options = data['options']
        if not isinstance(options, dict):
            return False, 'options must be a JSON object', 'INVALID_PARAMETER'
        
        if 'min_swatch_size' in options:
            if not isinstance(options['min_swatch_size'], int) or options['min_swatch_size'] < 1:
                return False, 'options.min_swatch_size must be a positive integer', 'INVALID_PARAMETER'
        
        if 'max_swatch_size' in options:
            if not isinstance(options['max_swatch_size'], int) or options['max_swatch_size'] < 1:
                return False, 'options.max_swatch_size must be a positive integer', 'INVALID_PARAMETER'
        
        if 'expected_rows' in options:
            if not isinstance(options['expected_rows'], int) or options['expected_rows'] < 1:
                return False, 'options.expected_rows must be a positive integer', 'INVALID_PARAMETER'
    
    return True, None, None


# Lazy-load swatch detection service
_swatch_detection_service = None

def get_swatch_detection_service():
    """Get or create swatch detection service."""
    global _swatch_detection_service
    if _swatch_detection_service is None:
        from services.pipeline.steps.bottle.reference_square_detection import SwatchDetectionService
        _swatch_detection_service = SwatchDetectionService()
    return _swatch_detection_service


@app.route('/detect-color-swatches', methods=['POST'])
@limiter.limit("15 per minute")  # Moderate operation
def detect_color_swatches():
    """
    Detect color swatches using pure OpenCV (no AI).
    
    Fast and free. Use after user has adjusted the image and wants
    to re-detect swatch positions.
    
    Request (JSON):
    - image_path: URL or path to the (already cropped/rotated) image
    - options: Optional detection parameters:
        - min_swatch_size: int (default: 15)
        - max_swatch_size: int (default: 150)
        - expected_rows: int (hint for clustering)
    
    Returns:
    - rows: Array of detected rows, each with swatches
    - total_swatches: Total number of swatches found
    - image_dimensions: Width and height of processed image
    - processing_time_ms: Processing time
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
        is_valid, error_msg, error_code = validate_detect_swatches_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': error_code
            }), 400
        
        image_path = data['image_path']
        options = data.get('options', {})
        
        logger.info(f'Detecting swatches in: {image_path}')
        
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
        
        # Detect swatches
        swatch_service = get_swatch_detection_service()
        result = swatch_service.detect_swatches(image, options)
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Detection failed'),
                'error_code': result.get('error_code', 'DETECTION_ERROR')
            }), 500
        
        return jsonify(result)
        
    except Exception as e:
        request_id = getattr(g, 'request_id', 'unknown')
        logger.error(f'[Request {request_id}] Error in detect_color_swatches: {str(e)}', exc_info=True)
        return jsonify({
            'success': False,
            'error': sanitize_error_message(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f'Starting PoolGuy CV Service on port {port}')
    app.run(host='0.0.0.0', port=port, debug=debug)










