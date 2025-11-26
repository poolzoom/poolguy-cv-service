"""
OpenAI Vision service for PoolGuy CV Service.
Uses GPT-4o vision API to detect test strip and pad locations.
"""

import cv2
import numpy as np
import logging
import base64
import json
import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. Install with: pip install openai>=1.0.0")

logger = logging.getLogger(__name__)


class OpenAIVisionService:
    """Service for detecting test strip and pads using OpenAI GPT-4o vision API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        image_detail: str = "high"
    ):
        """
        Initialize OpenAI vision service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            timeout: Request timeout in seconds
            model: Model to use (default: "gpt-4o", options: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo")
            temperature: Sampling temperature (0.0 = deterministic, 0.0-2.0 range)
            max_tokens: Maximum tokens in response
            image_detail: Image detail level ("low", "high", or "auto")
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_detail = image_detail
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Install with: pip install openai>=1.0.0")
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI Vision Service initialized")
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode numpy image to base64 string for API.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Base64-encoded PNG string
        """
        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode as PNG
        success, buffer = cv2.imencode('.png', rgb_image)
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _make_api_call(
        self,
        image: np.ndarray,
        prompt: str
    ) -> Optional[Dict]:
        """
        Make API call to OpenAI.
        
        Args:
            image: Input image
            prompt: Prompt text
            
        Returns:
            API response dict or None if failed
        """
        image_base64 = self._encode_image(image)
        h, w = image.shape[:2]
        
        try:
            # Log the request (prompt)
            self.logger.info("=== OpenAI API Request ===")
            self.logger.info(f"Prompt: {prompt}")
            self.logger.info(f"Image size: {w}x{h}, Base64 length: {len(image_base64)}")
            
            # Log API parameters
            self.logger.info(f"Model: {self.model}, Temperature: {self.temperature}, Max tokens: {self.max_tokens}, Image detail: {self.image_detail}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": self.image_detail
                                }
                            }
                        ]
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=self.timeout
            )
            
            # Log the raw response
            content = response.choices[0].message.content
            self.logger.info("=== OpenAI API Raw Response ===")
            self.logger.info(f"Raw JSON (full):\n{content}")
            
            result = json.loads(content)
            self.logger.info(f"Parsed result:\n{json.dumps(result, indent=2)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            return None
    
    def detect_strip(self, image: np.ndarray) -> Dict:
        """
        Detect test strip location using OpenAI vision.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with strip detection results:
            {
                'success': bool,
                'strip_region': {'top': int, 'bottom': int, 'left': int, 'right': int},
                'orientation': str,
                'angle': float,
                'confidence': float,
                'method': 'openai'
            }
        """
        h, w = image.shape[:2]
        
        prompt = f"""You are analyzing an image to locate a pool test strip. The test strip is a long, narrow white rectangle with colored square pads arranged in a line.

CRITICAL: Image dimensions are {w} x {h} pixels (width x height). All coordinates must be within these bounds.

The test strip characteristics:
- Long and narrow (aspect ratio typically 5:1 to 10:1)
- Usually positioned near the center of the image
- White background with colored square pads
- May be slightly rotated (typically within ±5 degrees)

TASK: Identify the EXACT pixel coordinates of the test strip's bounding box.

IMPORTANT:
- Be precise with coordinates - measure carefully
- The strip is usually 50-80% of the image height (if vertical) or width (if horizontal)
- Top-left corner of image is (0, 0)
- Coordinates are: top (y-min), bottom (y-max), left (x-min), right (x-max)
- All coordinates must be integers between 0 and image dimensions

Return your response as JSON with this exact format:
{{
    "success": true,
    "strip_region": {{
        "top": <integer between 0 and {h}>,
        "bottom": <integer between 0 and {h}, must be > top>,
        "left": <integer between 0 and {w}>,
        "right": <integer between 0 and {w}, must be > left>
    }},
    "orientation": "vertical" or "horizontal",
    "angle": <float, typically -5 to +5 degrees>,
    "confidence": <float between 0 and 1>
}}

If you cannot find a test strip, return:
{{
    "success": false,
    "error": "Could not detect test strip"
}}
"""
        
        try:
            result = self._make_api_call(image, prompt)
            
            if not result:
                return {
                    'success': False,
                    'error': 'OpenAI API call failed',
                    'method': 'openai'
                }
            
            # Validate response format
            if not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'method': 'openai'
                }
            
            # Validate coordinates
            strip_region = result.get('strip_region', {})
            if not all(key in strip_region for key in ['top', 'bottom', 'left', 'right']):
                return {
                    'success': False,
                    'error': 'Invalid strip region format',
                    'method': 'openai'
                }
            
            # Validate coordinate ranges
            top = int(strip_region['top'])
            bottom = int(strip_region['bottom'])
            left = int(strip_region['left'])
            right = int(strip_region['right'])
            
            if not (0 <= top < bottom <= h and 0 <= left < right <= w):
                return {
                    'success': False,
                    'error': f'Invalid coordinates: top={top}, bottom={bottom}, left={left}, right={right}',
                    'method': 'openai'
                }
            
            # Return validated result
            return {
                'success': True,
                'strip_region': {
                    'top': top,
                    'bottom': bottom,
                    'left': left,
                    'right': right
                },
                'orientation': result.get('orientation', 'vertical'),
                'angle': float(result.get('angle', 0.0)),
                'confidence': float(result.get('confidence', 0.8)),
                'method': 'openai'
            }
            
        except Exception as e:
            self.logger.error(f"Error in detect_strip: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'method': 'openai'
            }
    
    def detect_pads(
        self,
        image: np.ndarray,
        strip_region: Dict
    ) -> List[Dict]:
        """
        Detect pad locations within a detected strip using OpenAI vision.
        
        Args:
            image: Input image (BGR format)
            strip_region: Strip region dict with top, bottom, left, right
            
        Returns:
            List of detected pad regions:
            [
                {
                    'pad_index': int,
                    'x': int,
                    'y': int,
                    'width': int,
                    'height': int,
                    'confidence': float,
                    'method': 'openai'
                },
                ...
            ]
        """
        h, w = image.shape[:2]
        
        # Crop image to strip region to reduce tokens/cost
        top = max(0, strip_region['top'])
        bottom = min(h, strip_region['bottom'])
        left = max(0, strip_region['left'])
        right = min(w, strip_region['right'])
        
        strip_roi = image[top:bottom, left:right]
        if strip_roi.size == 0:
            return []
        
        roi_h, roi_w = strip_roi.shape[:2]
        
        prompt = f"""Analyze this cropped image of a pool test strip. The strip contains colored square pads arranged in a line.

Image dimensions: {roi_w} x {roi_h} pixels (width x height).

Please identify:
1. The number of color pads visible
2. The bounding box for each pad (x, y, width, height in pixels relative to this cropped image)
3. Order the pads from top to bottom (or left to right if horizontal)

Return your response as JSON with this exact format:
{{
    "pad_count": <integer>,
    "pads": [
        {{
            "pad_index": <integer starting from 0>,
            "x": <integer>,
            "y": <integer>,
            "width": <integer>,
            "height": <integer>,
            "confidence": <float between 0 and 1>
        }},
        ...
    ]
}}

Coordinates are relative to the cropped image (top-left is 0,0).
"""
        
        try:
            result = self._make_api_call(strip_roi, prompt)
            
            if not result:
                return []
            
            pads = result.get('pads', [])
            if not pads:
                return []
            
            # Convert pad coordinates from ROI space to original image space
            detected_pads = []
            for pad in pads:
                # Validate pad data
                if not all(key in pad for key in ['pad_index', 'x', 'y', 'width', 'height']):
                    continue
                
                # Get coordinates in ROI space
                pad_x = int(pad['x'])
                pad_y = int(pad['y'])
                pad_w = int(pad['width'])
                pad_h = int(pad['height'])
                
                # Validate ROI coordinates
                if not (0 <= pad_x < roi_w and 0 <= pad_y < roi_h and pad_w > 0 and pad_h > 0):
                    continue
                
                # Convert to original image coordinates
                orig_x = left + pad_x
                orig_y = top + pad_y
                
                # Validate original coordinates
                if not (0 <= orig_x < w and 0 <= orig_y < h):
                    continue
                
                detected_pads.append({
                    'pad_index': int(pad['pad_index']),
                    'x': orig_x,
                    'y': orig_y,
                    'width': pad_w,
                    'height': pad_h,
                    'confidence': float(pad.get('confidence', 0.8)),
                    'method': 'openai'
                })
            
            # Sort by pad_index
            detected_pads.sort(key=lambda p: p['pad_index'])
            
            return detected_pads
            
        except Exception as e:
            self.logger.error(f"Error in detect_pads: {e}", exc_info=True)
            return []

