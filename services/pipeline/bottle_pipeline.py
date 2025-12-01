"""
Bottle Label Analysis Service for PoolGuy CV Service.

Analyzes test strip bottle label images to detect:
- Brand/product name
- Test parameters (pads)
- Color swatches with values
- LAB colors for each swatch

This is a stateless service - each request is independent.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional

from services.utils.debug import DebugContext
from utils.image_loader import load_image
from utils.color_conversion import rgb_to_hex

logger = logging.getLogger(__name__)


# Validation limits
MAX_PADS = 8  # Maximum number of pads on a test strip bottle
MAX_SWATCHES_PER_PAD = 10  # Maximum swatches per pad

# Parameter name normalization mapping
PARAMETER_MAPPINGS = {
    'free chlorine': 'chlorine',
    'chlorine': 'chlorine',
    'fcl': 'chlorine',
    'cl': 'chlorine',
    'total chlorine': 'total_chlorine',
    'tcl': 'total_chlorine',
    'ph': 'ph',
    'alkalinity': 'alkalinity',
    'total alkalinity': 'alkalinity',
    'alk': 'alkalinity',
    'hardness': 'hardness',
    'total hardness': 'hardness',
    'calcium': 'calcium',
    'calcium hardness': 'calcium',
    'cya': 'cya',
    'cyanuric acid': 'cya',
    'stabilizer': 'cya',
    'bromine': 'bromine',
    'br': 'bromine',
}


class BottlePipelineService:
    """
    Service for analyzing test strip bottle label images.
    
    Uses OpenAI Vision for understanding (brand, parameters, values)
    and OpenCV for precision (color extraction).
    """
    
    def __init__(self):
        """Initialize bottle pipeline service."""
        self.logger = logging.getLogger(__name__)
        self._text_extraction_service = None
    
    @property
    def text_extraction_service(self):
        """Lazy load text extraction service."""
        if self._text_extraction_service is None:
            from services.pipeline.steps.bottle.text_extraction import TextExtractionService
            self._text_extraction_service = TextExtractionService()
        return self._text_extraction_service
    
    def analyze_bottle_label(
        self,
        image_path: str,
        hints: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Dict:
        """
        Analyze a bottle label image.
        
        Args:
            image_path: URL or path to the bottle image
            hints: Optional hints to improve detection:
                - expected_pad_count: int
                - known_brand: str
                - known_parameters: List[str]
            debug: Optional debug context for visualization
        
        Returns:
            Analysis result with brand, pads, and swatches
        """
        start_time = time.time()
        hints = hints or {}
        
        self.logger.info(f"Analyzing bottle label: {image_path}")
        
        # Step 1: Load image
        try:
            image = load_image(image_path)
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return self._error_result(f'Failed to load image: {str(e)}', 'IMAGE_LOAD_ERROR')
        
        if image is None:
            return self._error_result('Image loaded as None', 'IMAGE_LOAD_ERROR')
        
        h, w = image.shape[:2]
        self.logger.info(f"Image loaded: {w}x{h} pixels")
        
        # Debug: Input image
        if debug:
            debug.add_step('00_input', 'Input Image', image.copy(), {
                'image_path': image_path,
                'width': w,
                'height': h,
                'hints': hints
            })
        
        # Step 2: Extract text and structure using OpenAI Vision
        self.logger.info("Extracting text and structure with OpenAI Vision...")
        
        text_result = self.text_extraction_service.extract_text(
            image=image,
            method='openai'
        )
        
        if not text_result.get('success'):
            error_msg = text_result.get('error', 'Text extraction failed')
            self.logger.error(f"Text extraction failed: {error_msg}")
            
            if debug:
                debug.add_step('01_extraction_failed', 'Extraction Failed', image.copy(), {
                    'error': error_msg
                })
            
            return self._error_result(error_msg, 'OPENAI_ERROR')
        
        pad_names = text_result.get('pad_names', [])
        raw_text = text_result.get('raw_text', '')
        
        self.logger.info(f"Found {len(pad_names)} pads")
        
        # Validate pad count - too many pads indicates detection error
        if len(pad_names) > MAX_PADS:
            error_msg = f"Detection error: found {len(pad_names)} pads, maximum is {MAX_PADS}"
            self.logger.error(error_msg)
            return self._error_result(error_msg, 'TOO_MANY_PADS')
        
        # Validate swatch count per pad
        for i, pad in enumerate(pad_names):
            swatches = pad.get('reference_values', [])
            if len(swatches) > MAX_SWATCHES_PER_PAD:
                error_msg = f"Detection error: pad {i} has {len(swatches)} swatches, maximum is {MAX_SWATCHES_PER_PAD}"
                self.logger.error(error_msg)
                return self._error_result(error_msg, 'TOO_MANY_SWATCHES')
        
        # Debug: Text extraction
        if debug:
            debug.add_step('01_text_extraction', 'Text Extraction', image.copy(), {
                'pads_found': len(pad_names),
                'raw_text_preview': raw_text[:200] if raw_text else ''
            })
        
        # Step 3: Build response in new format
        result = self._build_response(
            image=image,
            pad_names=pad_names,
            raw_text=raw_text,
            start_time=start_time
        )
        
        # Debug: Final result
        if debug:
            vis = self._visualize_result(image, result)
            debug.add_step('02_final_result', 'Final Result', vis, {
                'pads_detected': len(result['data']['pads']),
                'brand': result['data']['brand']['name'] if result['data']['brand']['name'] else 'Unknown'
            })
        
        return result
    
    def _build_response(
        self,
        image: np.ndarray,
        pad_names: List[Dict],
        raw_text: str,
        start_time: float
    ) -> Dict:
        """
        Build the response in the new simplified format.
        
        Args:
            image: Original image
            pad_names: Extracted pad data from text extraction
            raw_text: Raw text detected in image
            start_time: Processing start time
        
        Returns:
            Formatted response dict
        """
        h, w = image.shape[:2]
        
        # Extract brand from raw text
        brand_info = self._extract_brand_info(raw_text)
        
        # Build pads array
        pads = []
        for i, pad_data in enumerate(pad_names):
            pad = self._format_pad(i, pad_data)
            pads.append(pad)
        
        # Calculate overall detection confidence
        if pads:
            avg_confidence = sum(p['confidence'] for p in pads) / len(pads)
        else:
            avg_confidence = 0.0
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'data': {
                'detection': {
                    'is_test_strip_bottle': len(pads) > 0,
                    'confidence': avg_confidence
                },
                'brand': brand_info,
                'pads': pads,
                'image_dimensions': {
                    'width': w,
                    'height': h
                },
                'processing_time_ms': processing_time_ms
            }
        }
    
    def _extract_brand_info(self, raw_text: str) -> Dict:
        """
        Extract brand information from raw text.
        
        Args:
            raw_text: Raw text detected in image
        
        Returns:
            Brand info dict
        """
        # Common brand names to look for
        known_brands = [
            'AquaChek', 'Taylor', 'HTH', 'Clorox', 'In The Swim',
            'Pool Master', 'LaMotte', 'Hach', 'Poolmaster'
        ]
        
        detected_text = []
        brand_name = None
        confidence = 0.0
        
        if raw_text:
            # Split into words/phrases
            words = raw_text.replace('\n', ' ').split()
            detected_text = [w for w in words if len(w) > 2][:20]  # First 20 significant words
            
            # Look for known brands
            text_lower = raw_text.lower()
            for brand in known_brands:
                if brand.lower() in text_lower:
                    brand_name = brand
                    confidence = 0.9
                    break
            
            # Look for common product patterns
            if not brand_name:
                if '7-way' in text_lower or '7 way' in text_lower:
                    brand_name = 'AquaChek 7-Way'
                    confidence = 0.7
                elif '6-in-1' in text_lower or '6 in 1' in text_lower:
                    brand_name = 'AquaChek 6-in-1'
                    confidence = 0.7
                elif '4-in-1' in text_lower or '4 in 1' in text_lower:
                    brand_name = 'AquaChek 4-in-1'
                    confidence = 0.7
        
        return {
            'name': brand_name,
            'confidence': confidence,
            'detected_text': detected_text
        }
    
    def _format_pad(self, index: int, pad_data: Dict) -> Dict:
        """
        Format a single pad in the new response format.
        
        Args:
            index: Pad index
            pad_data: Raw pad data from text extraction
        
        Returns:
            Formatted pad dict
        """
        name = pad_data.get('name', f'Pad {index + 1}')
        
        # Normalize parameter name
        parameter = self._normalize_parameter(name)
        
        # Get pad region
        pad_region = pad_data.get('pad_region', pad_data.get('region', {}))
        
        # Build swatches from reference_values
        swatches = []
        reference_values = pad_data.get('reference_values', [])
        
        for j, ref_val in enumerate(reference_values):
            swatch = self._format_swatch(j, ref_val)
            swatches.append(swatch)
        
        # Calculate confidence
        confidence = 0.5
        if name and name != f'Pad {index + 1}':
            confidence += 0.2
        if pad_data.get('reference_range'):
            confidence += 0.15
        if len(swatches) > 0:
            confidence += 0.15
        
        return {
            'pad_index': index,
            'parameter': parameter,
            'display_name': name,
            'confidence': min(confidence, 1.0),
            'region': {
                'x': int(pad_region.get('x', 0)),
                'y': int(pad_region.get('y', 0)),
                'width': int(pad_region.get('width', 0)),
                'height': int(pad_region.get('height', 0))
            } if pad_region else None,
            'swatches': swatches
        }
    
    def _format_swatch(self, index: int, ref_val: Dict) -> Dict:
        """
        Format a single swatch in the new response format.
        
        Args:
            index: Swatch index
            ref_val: Reference value data
        
        Returns:
            Formatted swatch dict
        """
        region = ref_val.get('region', {})
        lab_color = ref_val.get('lab_color', {})
        
        # Convert LAB to hex if we have LAB values
        hex_color = None
        if lab_color and 'L' in lab_color:
            # Approximate conversion (LAB -> RGB -> Hex)
            # This is simplified; actual conversion is more complex
            L = lab_color.get('L', 50)
            a = lab_color.get('a', 0)
            b = lab_color.get('b', 0)
            
            # Simple approximation for display purposes
            # Real conversion would use proper LAB->RGB formulas
            r = int(min(255, max(0, L * 2.55 + a)))
            g = int(min(255, max(0, L * 2.55 - a/2 - b/2)))
            b_val = int(min(255, max(0, L * 2.55 - b)))
            hex_color = rgb_to_hex(r, g, b_val)
        
        return {
            'swatch_index': index,
            'value': str(ref_val.get('value', '')),
            'region': {
                'x': int(region.get('x', 0)),
                'y': int(region.get('y', 0)),
                'width': int(region.get('width', 0)),
                'height': int(region.get('height', 0))
            } if region else None,
            'color': {
                'lab': {
                    'L': round(lab_color.get('L', 0), 2),
                    'a': round(lab_color.get('a', 0), 2),
                    'b': round(lab_color.get('b', 0), 2)
                } if lab_color else None,
                'hex': hex_color
            }
        }
    
    def _normalize_parameter(self, name: str) -> str:
        """
        Normalize a parameter name to a standard key.
        
        Args:
            name: Display name (e.g., "Free Chlorine")
        
        Returns:
            Normalized key (e.g., "chlorine")
        """
        if not name:
            return 'unknown'
        
        name_lower = name.lower().strip()
        
        # Check exact matches first
        if name_lower in PARAMETER_MAPPINGS:
            return PARAMETER_MAPPINGS[name_lower]
        
        # Check partial matches
        for key, value in PARAMETER_MAPPINGS.items():
            if key in name_lower:
                return value
        
        return 'unknown'
    
    def _visualize_result(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        Create visualization of the analysis result.
        
        Args:
            image: Original image
            result: Analysis result
        
        Returns:
            Visualization image
        """
        import cv2
        
        vis = image.copy()
        data = result.get('data', {})
        pads = data.get('pads', [])
        
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 128, 255),# Light red
        ]
        
        for i, pad in enumerate(pads):
            color = colors[i % len(colors)]
            
            # Draw pad region
            region = pad.get('region')
            if region and region.get('width', 0) > 0:
                x, y = region['x'], region['y']
                w, h = region['width'], region['height']
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                
                # Label
                label = f"{i+1}: {pad.get('display_name', 'Unknown')}"
                cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw swatches
            for swatch in pad.get('swatches', []):
                s_region = swatch.get('region')
                if s_region and s_region.get('width', 0) > 0:
                    sx, sy = s_region['x'], s_region['y']
                    sw, sh = s_region['width'], s_region['height']
                    cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), color, 1)
        
        # Add header
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 50), (0, 0, 0), -1)
        
        brand = data.get('brand', {}).get('name') or 'Unknown Brand'
        cv2.putText(vis, f'Brand: {brand}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f'Pads: {len(pads)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return vis
    
    def _error_result(self, error: str, error_code: str) -> Dict:
        """Create standardized error result."""
        return {
            'success': False,
            'error': error,
            'error_code': error_code
        }
    
    # Legacy method for backwards compatibility with existing endpoint
    def process_bottle(
        self,
        image_paths: List[str],
        options: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Dict:
        """
        Legacy method - wraps analyze_bottle_label for backwards compatibility.
        
        Args:
            image_paths: List of image paths (uses first one)
            options: Processing options
            debug: Optional debug context
        
        Returns:
            Legacy format result
        """
        if not image_paths:
            return {
                'success': False,
                'error': 'No image paths provided',
                'error_code': 'NO_IMAGES',
                'pads': [],
                'overall_confidence': 0.0,
                'images_processed': 0,
                'pads_detected': 0
            }
        
        # Use first image
        result = self.analyze_bottle_label(
            image_path=image_paths[0],
            hints=options,
            debug=debug
        )
        
        if not result.get('success'):
            return {
                'success': False,
                'error': result.get('error', 'Analysis failed'),
                'error_code': result.get('error_code', 'ANALYSIS_FAILED'),
                'pads': [],
                'overall_confidence': 0.0,
                'images_processed': 0,
                'pads_detected': 0
            }
        
        # Convert to legacy format
        data = result.get('data', {})
        pads = data.get('pads', [])
        
        legacy_pads = []
        for pad in pads:
            legacy_pad = {
                'pad_index': pad['pad_index'],
                'name': pad['display_name'],
                'region': pad.get('region'),
                'reference_range': None,  # Not in new format
                'reference_squares': [
                    {
                        'color': s['color']['lab'],
                        'value': s['value'],
                        'region': s['region']
                    }
                    for s in pad.get('swatches', [])
                ],
                'detected_color': None,
                'mapped_value': None,
                'confidence': pad['confidence']
            }
            legacy_pads.append(legacy_pad)
        
        return {
            'success': True,
            'pads': legacy_pads,
            'overall_confidence': data.get('detection', {}).get('confidence', 0.0),
            'images_processed': 1,
            'pads_detected': len(pads)
        }
