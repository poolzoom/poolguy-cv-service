"""
Text extraction service for bottle pipeline.

Extracts pad names and reference ranges using OCR.
Primary: OpenAI Vision API (accurate, understands context)
Fallback: Tesseract OCR (local, faster but less accurate)

IMPORTANT: No silent fallbacks. If the requested method fails, return failure.
"""

import cv2
import numpy as np
import logging
import json
from typing import Dict, List, Optional
import re

# Check for Tesseract availability
try:
    import pytesseract
    # Also check if tesseract binary is available
    try:
        pytesseract.get_tesseract_version()
        TESSERACT_AVAILABLE = True
    except pytesseract.TesseractNotFoundError:
        TESSERACT_AVAILABLE = False
        logging.warning("Tesseract binary not found. Install with: brew install tesseract")
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available. Install with: pip install pytesseract")

logger = logging.getLogger(__name__)


class TextExtractionService:
    """
    Service for extracting text from bottle images.
    
    Uses OpenAI Vision as primary method for accurate, context-aware extraction.
    Tesseract available as explicit fallback option.
    
    NO SILENT FALLBACKS: If a method fails, we report the failure.
    """
    
    def __init__(self, openai_service=None):
        """
        Initialize text extraction service.
        
        Args:
            openai_service: Optional pre-initialized OpenAI Vision service
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI service
        self._openai_service = openai_service
        self._openai_initialized = False
    
    @property
    def openai_service(self):
        """Lazy load OpenAI service with settings optimized for text extraction."""
        if not self._openai_initialized:
            if self._openai_service is None:
                try:
                    from services.detection.openai_vision import OpenAIVisionService
                    # Optimized for speed + quality: gpt-4o-mini + auto detail
                    # Benchmarks: ~6s with good accuracy (vs 30s with gpt-4o)
                    self._openai_service = OpenAIVisionService(
                        model="gpt-4o-mini",  # Faster model (5x faster than gpt-4o)
                        max_tokens=2000,
                        timeout=60,
                        image_detail="auto"  # Auto for best quality/speed balance
                    )
                except Exception as e:
                    self.logger.warning(f"OpenAI Vision not available: {e}")
                    self._openai_service = None
            self._openai_initialized = True
        return self._openai_service
    
    def extract_text(
        self,
        image: np.ndarray,
        method: str = "openai"
    ) -> Dict:
        """
        Extract text from bottle image.
        
        Args:
            image: Input image (BGR format)
            method: OCR method to use:
                - "openai": Use OpenAI Vision API (recommended)
                - "tesseract": Use local Tesseract OCR
        
        Returns:
            Dictionary with extraction result:
            {
                'success': bool,
                'pad_names': [...],
                'raw_text': str,
                'method': str,
                'error': str (only if failed)
            }
        """
        self.logger.info(f"Text extraction requested with method: {method}")
        
        # Route to appropriate method
        if method == "openai":
            return self._extract_with_openai(image)
        elif method == "tesseract":
            return self._extract_with_tesseract(image)
        else:
            return {
                'success': False,
                'error': f"Unknown OCR method: {method}. Use 'openai' or 'tesseract'.",
                'error_code': 'INVALID_OCR_METHOD',
                'pad_names': [],
                'raw_text': '',
                'method': method
            }
    
    def _extract_with_openai(self, image: np.ndarray) -> Dict:
        """
        Extract text using OpenAI Vision API.
        
        This is the preferred method as it:
        - Understands context (knows what pad names look like)
        - Can extract reference ranges associated with pads
        - Provides accurate bounding boxes
        - Can extract LAB colors from regions
        """
        # Check if OpenAI service is available
        if self.openai_service is None:
            return {
                'success': False,
                'error': 'OpenAI Vision service not available. Check OPENAI_API_KEY.',
                'error_code': 'OPENAI_NOT_AVAILABLE',
                'pad_names': [],
                'raw_text': '',
                'method': 'openai'
            }
        
        h, w = image.shape[:2]
        
        prompt = self._build_openai_prompt(w, h)
        
        try:
            self.logger.info(f"Making OpenAI API call for text extraction ({w}x{h} image)...")
            result = self.openai_service._make_api_call(image, prompt)
            
            if result is None:
                self.logger.error("OpenAI API returned None")
                return {
                    'success': False,
                    'error': 'OpenAI API call returned no result',
                    'error_code': 'OPENAI_NO_RESPONSE',
                    'pad_names': [],
                    'raw_text': '',
                    'method': 'openai'
                }
            
            # Log the response for debugging
            self.logger.info(f"OpenAI response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            self.logger.debug(f"OpenAI full response: {json.dumps(result, indent=2)}")
            
            # Check for success in response
            if not result.get('success', False):
                error = result.get('error', 'Unknown error from OpenAI')
                self.logger.warning(f"OpenAI returned success=False: {error}")
                return {
                    'success': False,
                    'error': f'OpenAI extraction failed: {error}',
                    'error_code': 'OPENAI_EXTRACTION_FAILED',
                    'pad_names': [],
                    'raw_text': result.get('raw_text', ''),
                    'method': 'openai'
                }
            
            # Process and validate the response
            validated_pads = self._validate_openai_response(result, image)
            raw_text = result.get('raw_text', '')
            
            self.logger.info(f"OpenAI extraction successful: {len(validated_pads)} pads found")
            
            return {
                'success': len(validated_pads) > 0,
                'error': None if len(validated_pads) > 0 else 'No valid pad names found in response',
                'pad_names': validated_pads,
                'raw_text': raw_text,
                'method': 'openai'
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI extraction exception: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'OpenAI extraction failed: {str(e)}',
                'error_code': 'OPENAI_EXCEPTION',
                'pad_names': [],
                'raw_text': '',
                'method': 'openai'
            }
    
    def _build_openai_prompt(self, width: int, height: int) -> str:
        """Build the OpenAI Vision prompt for text extraction."""
        # Simplified prompt - no coordinates (they're inaccurate anyway)
        # Focus on semantics: pad names and reference values
        return """Analyze this pool test strip bottle label. Extract the test parameters and their reference values.

Return a JSON object:
{
    "success": true,
    "pad_names": [
        {
            "name": "Free Chlorine",
            "reference_range": "0, 0.5, 1, 2, 3, 5, 10"
        },
        {
            "name": "pH", 
            "reference_range": "6.2, 6.8, 7.2, 7.8, 8.4, 9.0"
        }
    ],
    "raw_text": "All visible text from the label"
}

Include ALL test parameters shown (pH, Free Chlorine, Total Alkalinity, Cyanuric Acid, etc).
Include ALL reference values for each parameter."""
    
    def _validate_openai_response(
        self,
        result: Dict,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Validate OpenAI response - simplified version without color extraction.
        
        Color extraction happens later via /extract-colors-at-locations when
        the user provides correct positions.
        
        Args:
            result: Raw OpenAI response
            image: Original image (not used for colors anymore)
        
        Returns:
            List of validated pad dictionaries with names and reference values
        """
        # Validation limits
        MAX_PADS = 8
        
        pad_names = result.get('pad_names', [])
        
        # Early validation: reject unreasonable pad counts
        if len(pad_names) > MAX_PADS:
            self.logger.warning(f"OpenAI returned {len(pad_names)} pads, truncating to {MAX_PADS}")
            pad_names = pad_names[:MAX_PADS]
        
        validated_pads = []
        
        for pad in pad_names:
            if not isinstance(pad, dict):
                continue
            
            name = pad.get('name', '').strip()
            if not name:
                continue
            
            # Parse reference_range into individual values
            reference_range = pad.get('reference_range', '').strip()
            reference_values = []
            
            if reference_range:
                # Parse comma-separated values
                for val in reference_range.split(','):
                    val = val.strip()
                    if val:
                        reference_values.append({'value': val})
            
            validated_pads.append({
                'name': name,
                'reference_range': reference_range or None,
                'reference_values': reference_values
            })
        
        return validated_pads
    
    def _extract_lab_from_region(
        self,
        lab_image: np.ndarray,
        region: Dict,
        img_width: int,
        img_height: int,
        fallback: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Extract LAB color from a region of the image.
        
        Args:
            lab_image: Image in LAB color space
            region: Region dict with x, y, width, height
            img_width: Image width for bounds checking
            img_height: Image height for bounds checking
            fallback: Fallback LAB values if extraction fails
        
        Returns:
            LAB color dict or None
        """
        if not isinstance(region, dict):
            return self._parse_lab_fallback(fallback)
        
        x = int(region.get('x', 0))
        y = int(region.get('y', 0))
        w = int(region.get('width', 0))
        h = int(region.get('height', 0))
        
        # Validate coordinates
        if w <= 0 or h <= 0:
            return self._parse_lab_fallback(fallback)
        
        if x < 0 or y < 0 or x >= img_width or y >= img_height:
            return self._parse_lab_fallback(fallback)
        
        # Clamp to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        try:
            from utils.color_conversion import extract_lab_values
            lab_dict = extract_lab_values(lab_image, (x, y, w, h))
            return {
                'L': lab_dict['L'],
                'a': lab_dict['a'],
                'b': lab_dict['b']
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract LAB: {e}")
            return self._parse_lab_fallback(fallback)
    
    def _parse_lab_fallback(self, fallback: Optional[Dict]) -> Optional[Dict]:
        """Parse fallback LAB color from API response."""
        if not isinstance(fallback, dict):
            return None
        if 'L' not in fallback:
            return None
        return {
            'L': float(fallback.get('L', 0)),
            'a': float(fallback.get('a', 0)),
            'b': float(fallback.get('b', 0))
        }
    
    def _extract_with_tesseract(self, image: np.ndarray) -> Dict:
        """
        Extract text using Tesseract OCR.
        
        Note: Tesseract is less accurate than OpenAI for this use case because:
        - It doesn't understand context (what pad names look like)
        - It cannot extract reference ranges or associate them with pads
        - It cannot identify pad regions or reference color squares
        
        Use only if OpenAI is unavailable or for cost savings.
        """
        if not TESSERACT_AVAILABLE:
            return {
                'success': False,
                'error': 'Tesseract not available. Install with: brew install tesseract',
                'error_code': 'TESSERACT_NOT_AVAILABLE',
                'pad_names': [],
                'raw_text': '',
                'method': 'tesseract'
            }
        
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Get raw text
            raw_text = pytesseract.image_to_string(thresh)
            
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            # Extract recognized words with positions
            pad_names = self._process_tesseract_data(data)
            
            self.logger.info(f"Tesseract found {len(pad_names)} potential pad names")
            
            return {
                'success': len(pad_names) > 0,
                'error': None if len(pad_names) > 0 else 'No pad names found',
                'pad_names': pad_names,
                'raw_text': raw_text,
                'method': 'tesseract'
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract extraction error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f'Tesseract extraction failed: {str(e)}',
                'error_code': 'TESSERACT_EXCEPTION',
                'pad_names': [],
                'raw_text': '',
                'method': 'tesseract'
            }
    
    def _process_tesseract_data(self, data: Dict) -> List[Dict]:
        """
        Process Tesseract output data to extract pad names.
        
        Note: Tesseract cannot provide:
        - Pad regions (colored squares)
        - Reference ranges
        - Reference color squares
        - LAB colors
        """
        pad_names = []
        
        # Common pad name patterns
        pad_patterns = [
            r'\bpH\b',
            r'\bChlorine\b',
            r'\bAlkalinity\b',
            r'\bCalcium\b',
            r'\bHardness\b',
            r'\bCyanuric\b',
            r'\bAcid\b',
            r'\bFCI\b',
            r'\bAlk\b',
            r'\bTotal\b',
            r'\bFree\b',
            r'\bBromine\b'
        ]
        
        n_boxes = len(data.get('text', []))
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Skip low confidence or empty
            if conf < 30 or not text:
                continue
            
            # Check if it matches a pad name pattern
            is_pad_name = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in pad_patterns
            )
            
            if is_pad_name:
                pad_names.append({
                    'name': text,
                    'region': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    },
                    'pad_region': None,  # Tesseract cannot detect this
                    'reference_range': None,  # Tesseract cannot extract this
                    'lab_color': None,  # Tesseract cannot extract this
                    'reference_values': []  # Tesseract cannot extract this
                })
        
        return pad_names
