"""
Color extraction service for PoolGuy CV Service.
Detects test strip pads and extracts LAB color values with confidence scores.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from utils.image_loader import load_image
from utils.color_conversion import (
    bgr_to_lab, 
    extract_lab_values_with_variance,
    normalize_white_balance
)
from services.utils.image_quality import ImageQualityService
from services.interfaces import PadRegion, ColorResult, ColorExtractionResult

logger = logging.getLogger(__name__)


class ColorExtractionService:
    """Service for extracting pad colors from test strip images."""
    
    def __init__(self):
        """Initialize color extraction service."""
        self.logger = logging.getLogger(__name__)
        self.quality_service = ImageQualityService()
    
    def extract_colors(
        self,
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        pad_regions: Optional[List[PadRegion]] = None,
        expected_pad_count: int = 6,
        normalize_white: bool = True,
        debug: Optional[Any] = None
    ) -> Dict:
        """
        Extract pad colors from test strip image.
        
        Args:
            image_path: Path to test strip image (local or URL). Required if image is None.
            image: Image array in BGR format. Required if image_path is None.
            pad_regions: Optional list of pre-detected pad regions. If provided, skips pad detection.
            expected_pad_count: Expected number of pads (4-7). Only used if pad_regions is None.
            normalize_white: Whether to apply white balance normalization
        
        Returns:
            Dictionary with extraction results:
            {
                'success': bool,
                'data': {
                    'pads': [...],
                    'overall_confidence': float
                }
            }
        """
        try:
            # Validate inputs
            if image_path is None and image is None:
                return {
                    'success': False,
                    'error': 'Either image_path or image must be provided',
                    'error_code': 'INVALID_PARAMETER'
                }
            
            # Validate pad count
            if not (3 <= expected_pad_count <= 7):
                return {
                    'success': False,
                    'error': f'expected_pad_count must be between 3 and 7, got {expected_pad_count}',
                    'error_code': 'INVALID_PARAMETER'
                }
            
            # Load image if needed
            if image is None:
                image = load_image(image_path)
            
            # Get image quality metrics for confidence calculation
            quality_metrics = self.quality_service._calculate_metrics(image)
            
            # Convert to LAB color space
            lab_image = bgr_to_lab(image)
            
            # Detect white regions between pads for normalization
            white_regions = []
            white_norm_success = False
            # NOTE: White balance normalization temporarily disabled for debugging/testing.
            # The normalization was causing color shifts that reduced accuracy in some cases.
            # Re-enable when normalization algorithm is improved or when testing shows it's needed.
            # Expected re-enable: After validation with production test strip images.
            # if normalize_white and pad_regions and len(pad_regions) > 1:
            #     white_regions = self._detect_white_regions_between_pads(
            #         lab_image, pad_regions, image
            #     )
            #     if white_regions:
            #         try:
            #             # Use detected white regions for normalization
            #             lab_image = self._normalize_white_balance_with_regions(
            #                 lab_image, white_regions
            #             )
            #             white_norm_success = True
            #             self.logger.debug(f'White balance normalization applied using {len(white_regions)} white regions')
            #         except Exception as e:
            #             self.logger.warning(f'White balance normalization failed: {e}')
            #     else:
            #         # Fallback to default method
            #         try:
            #             lab_image = normalize_white_balance(lab_image)
            #             white_norm_success = True
            #             self.logger.debug('White balance normalization applied (fallback method)')
            #         except Exception as e:
            #             self.logger.warning(f'White balance normalization failed: {e}')
            # elif normalize_white:
            #     # No pad regions, use default method
            #     try:
            #         lab_image = normalize_white_balance(lab_image)
            #         white_norm_success = True
            #         self.logger.debug('White balance normalization applied (default method)')
            #     except Exception as e:
            #         self.logger.warning(f'White balance normalization failed: {e}')
            
            # Debug: Visualize white reference regions
            if debug and white_regions:
                vis_white = image.copy()
                for i, white_region in enumerate(white_regions):
                    x, y, w, h = white_region
                    cv2.rectangle(vis_white, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(vis_white, f"White {i+1}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                debug.add_step('03_00_white_reference', 'White Reference Regions', vis_white, {
                    'white_region_count': len(white_regions),
                    'regions': [
                        {'x': x, 'y': y, 'width': w, 'height': h}
                        for x, y, w, h in white_regions
                    ]
                })
            
            # Get pad regions - either use provided ones or detect them
            if pad_regions is not None:
                # Use provided pad regions
                self.logger.info(f'Using {len(pad_regions)} pre-detected pad regions')
                # Convert PadRegion to (x, y, width, height) tuples for extraction
                pad_region_tuples = [
                    (pad['x'], pad['y'], pad['width'], pad['height'])
                    for pad in pad_regions
                ]
            else:
                # Detect pads using internal method
                self.logger.info(f'Detecting pads internally (expected: {expected_pad_count})')
                pad_region_tuples = self._detect_pads(image, expected_pad_count)
                
                if len(pad_region_tuples) != expected_pad_count:
                    return {
                        'success': False,
                        'error': f'Expected {expected_pad_count} pads, detected {len(pad_region_tuples)}',
                        'error_code': 'PAD_DETECTION_FAILED',
                        'detected_count': len(pad_region_tuples)
                    }
            
            # Extract colors from each pad
            pads = []
            inner_regions = []  # Store inner regions for debug visualization
            for idx, region in enumerate(pad_region_tuples):
                # Calculate inner region (exclude 15% from edges)
                inner_region = self._get_inner_region(region)
                inner_regions.append(inner_region)
                
                pad_data = self._extract_pad_color(
                    lab_image, 
                    inner_region,  # Use inner region instead of full region
                    idx,
                    quality_metrics,
                    white_norm_success,
                    original_region=region  # Keep original for coordinates
                )
                pads.append(pad_data)
            
            # Debug: Visualize reduced pad regions
            if debug:
                vis_inner = image.copy()
                for i, (outer, inner) in enumerate(zip(pad_region_tuples, inner_regions)):
                    # Draw outer region in light color
                    x_outer, y_outer, w_outer, h_outer = outer
                    cv2.rectangle(vis_inner, (x_outer, y_outer), 
                                 (x_outer + w_outer, y_outer + h_outer), (128, 128, 128), 1)
                    # Draw inner region in bright color
                    x_inner, y_inner, w_inner, h_inner = inner
                    cv2.rectangle(vis_inner, (x_inner, y_inner), 
                                 (x_inner + w_inner, y_inner + h_inner), (0, 255, 0), 2)
                    cv2.putText(vis_inner, f"P{i+1}", (x_inner, y_inner - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                debug.add_step('03_01_pad_inner_regions', 'Pad Inner Regions (Color Extraction)', vis_inner, {
                    'pad_count': len(pad_region_tuples),
                    'inner_regions': [
                        {'pad_index': i, 'x': x, 'y': y, 'width': w, 'height': h}
                        for i, (x, y, w, h) in enumerate(inner_regions)
                    ],
                    'outer_regions': [
                        {'pad_index': i, 'x': x, 'y': y, 'width': w, 'height': h}
                        for i, (x, y, w, h) in enumerate(pad_region_tuples)
                    ]
                })
            
            # Calculate overall confidence
            overall_confidence = np.mean([p['pad_detection_confidence'] for p in pads])
            
            return {
                'success': True,
                'data': {
                    'pads': pads,  # Each pad now includes 'region' with coordinates
                    'overall_confidence': float(overall_confidence)
                }
            }
            
        except ValueError as e:
            self.logger.error(f'Validation error: {e}')
            return {
                'success': False,
                'error': str(e),
                'error_code': 'INVALID_PARAMETER'
            }
        except Exception as e:
            self.logger.error(f'Error extracting colors: {e}', exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'COLOR_EXTRACTION_FAILED'
            }
    
    def _detect_pads(
        self, 
        image: np.ndarray, 
        expected_count: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect pad regions in test strip image.
        Multi-step pipeline with validation:
        1. Detect general strip area (bright vertical/horizontal region)
        2. Determine orientation (vertical vs horizontal)
        3. Detect top/bottom edges (for vertical) or left/right edges (for horizontal)
        4. Detect left/right edges accounting for lighting artifacts
        5. Find colored squares using known strip dimensions
        
        Args:
            image: OpenCV image array in BGR format
            expected_count: Expected number of pads
        
        Returns:
            List of (x, y, width, height) tuples for each pad region
        """
        h, w = image.shape[:2]
        
        # Step 1: Detect general strip area
        strip_region = self._detect_strip_general_area(image)
        if not strip_region:
            self.logger.warning('Step 1 failed: Could not detect general strip area')
            return []
        
        x_approx, y_approx, w_approx, h_approx = strip_region
        self.logger.info(f'Step 1: Detected approximate strip area: {w_approx}x{h_approx} at ({x_approx}, {y_approx})')
        
        # Step 1.5: Detect strip angle and determine orientation
        angle, is_vertical = self._detect_strip_angle_and_orientation(image, strip_region)
        if angle is None:
            self.logger.warning('Step 1.5 failed: Could not detect strip angle')
            return []
        
        self.logger.info(f'Step 1.5: Strip angle={angle:.2f}°, is_vertical={is_vertical}')
        
        # Step 2: Determine orientation and detect top/bottom edges (accounting for angle)
        top_edge, bottom_edge = self._detect_strip_vertical_edges_with_angle(
            image, strip_region, angle, is_vertical
        )
        if top_edge is None or bottom_edge is None:
            self.logger.warning('Step 2 failed: Could not detect top/bottom edges')
            return []
        
        self.logger.info(f'Step 2: Detected vertical edges: top={top_edge}, bottom={bottom_edge}')
        
        # Step 3: Detect left/right edges (accounting for lighting artifacts and angle)
        left_edge, right_edge = self._detect_strip_horizontal_edges_with_angle(
            image, x_approx, top_edge, bottom_edge, angle, is_vertical
        )
        if left_edge is None or right_edge is None:
            self.logger.warning('Step 3 failed: Could not detect left/right edges')
            return []
        
        self.logger.info(f'Step 3: Detected horizontal edges: left={left_edge}, right={right_edge}')
        
        # Step 3.5: Extract and rotate strip to align it
        strip_corners = self._get_strip_corners(left_edge, right_edge, top_edge, bottom_edge, angle)
        aligned_strip_image, transform_matrix = self._extract_and_align_strip(
            image, strip_corners
        )
        
        if aligned_strip_image is None:
            self.logger.warning('Step 3.5 failed: Could not extract and align strip')
            return []
        
        aligned_h, aligned_w = aligned_strip_image.shape[:2]
        self.logger.info(f'Step 3.5: Extracted aligned strip: {aligned_w}x{aligned_h}')
        
        # Step 4: Find colored squares within aligned strip
        pad_regions = self._detect_colored_squares_in_strip(
            aligned_strip_image, expected_count, aligned_w, aligned_h
        )
        
        if len(pad_regions) != expected_count:
            self.logger.warning(f'Step 4: Expected {expected_count} pads, found {len(pad_regions)}')
        
        # Step 5: Transform pad coordinates back to original image
        final_regions = self._transform_pad_coordinates_to_original(
            pad_regions, transform_matrix
        )
        
        # Sort by position (top to bottom)
        final_regions.sort(key=lambda r: r[1])
        
        self.logger.info(f'Final: Detected {len(final_regions)} pad regions')
        return final_regions
    
    def _detect_strip_general_area(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Step 1: Detect general strip area using projection method.
        Finds the brightest narrow vertical/horizontal band in the image.
        
        Returns:
            (x, y, width, height) of approximate strip region, or None
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal projection (sum of pixel values per column)
        projection = np.sum(gray, axis=0)
        
        # Smooth the projection
        window_size = max(10, int(w * 0.02))
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find the peak (brightest region = strip)
        peak_idx = np.argmax(smoothed)
        peak_value = smoothed[peak_idx]
        
        # Find width around peak (where values are above threshold)
        threshold = peak_value * 0.85
        
        # Find left and right boundaries
        left_bound = peak_idx
        right_bound = peak_idx
        
        for i in range(peak_idx, -1, -1):
            if smoothed[i] >= threshold:
                left_bound = i
            else:
                break
        
        for i in range(peak_idx, len(smoothed)):
            if smoothed[i] >= threshold:
                right_bound = i
            else:
                break
        
        strip_width = right_bound - left_bound
        
        # Validate width
        min_width = int(w * 0.03)   # At least 3% of image width
        max_width = int(w * 0.4)    # At most 40% of image width
        
        if min_width <= strip_width <= max_width:
            x = left_bound
            y = int(h * 0.05)  # Start 5% from top
            rh = int(h * 0.9)  # 90% of height
            
            return (x, y, strip_width, rh)
        
        return None
    
    def _detect_strip_angle_and_orientation(
        self,
        image: np.ndarray,
        strip_region: Tuple[int, int, int, int]
    ) -> Tuple[Optional[float], bool]:
        """
        Step 1.5: Detect the angle of the strip and determine if it's primarily vertical or horizontal.
        Uses multiple methods: Hough lines, edge orientation, and projection analysis.
        
        Returns:
            (angle, is_vertical)
            - angle: Rotation angle in degrees (0 = horizontal, 90 = vertical, -45 to 45 range)
            - is_vertical: True if strip is primarily vertical (angle > 45° or < -45°)
        """
        x_approx, y_approx, w_approx, h_approx = strip_region
        
        # Crop to approximate strip region with padding
        padding = 20
        x_start = max(0, x_approx - padding)
        y_start = max(0, y_approx - padding)
        x_end = min(image.shape[1], x_approx + w_approx + padding)
        y_end = min(image.shape[0], y_approx + h_approx + padding)
        
        roi = image[y_start:y_end, x_start:x_end]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray_roi.shape
        
        # Determine orientation first
        is_vertical = h_roi > w_roi * 1.5
        
        # Method 1: Hough lines for long edges
        edges = cv2.Canny(gray_roi, 50, 150)
        
        # Use HoughLinesP to detect long lines (strip edges)
        min_line_length = max(w_roi, h_roi) * 0.5  # At least 50% of dimension
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30, 
            minLineLength=int(min_line_length),
            maxLineGap=10
        )
        
        angles = []
        if lines is not None and len(lines) > 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Only consider lines that are roughly parallel to strip orientation
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                if is_vertical:
                    # For vertical strip, look for vertical lines (dy >> dx)
                    if dy > dx * 2:
                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg = np.degrees(angle_rad)
                        angles.append(angle_deg)
                else:
                    # For horizontal strip, look for horizontal lines (dx >> dy)
                    if dx > dy * 2:
                        angle_rad = np.arctan2(y2 - y1, x2 - x1)
                        angle_deg = np.degrees(angle_rad)
                        angles.append(angle_deg)
        
        # Method 2: Use projection to find angle
        # Rotate image at different angles and find which gives best projection peak
        if len(angles) < 3:  # Not enough lines found
            best_angle = 0.0
            best_peak_ratio = 0.0
            
            # Test angles from -10 to +10 degrees
            for test_angle in np.arange(-10, 11, 0.5):
                center = (w_roi // 2, h_roi // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, test_angle, 1.0)
                rotated = cv2.warpAffine(gray_roi, rotation_matrix, (w_roi, h_roi))
                
                if is_vertical:
                    # For vertical, use horizontal projection
                    projection = np.sum(rotated, axis=0)
                else:
                    # For horizontal, use vertical projection
                    projection = np.sum(rotated, axis=1)
                
                # Find peak sharpness (ratio of peak to mean)
                peak = np.max(projection)
                mean_val = np.mean(projection)
                peak_ratio = peak / mean_val if mean_val > 0 else 0
                
                if peak_ratio > best_peak_ratio:
                    best_peak_ratio = peak_ratio
                    best_angle = test_angle
            
            if best_peak_ratio > 1.2:  # Significant peak
                angles.append(best_angle)
        
        if len(angles) == 0:
            # Fallback: assume no rotation
            return (0.0, is_vertical)
        
        # Normalize angles to -90 to 90 range
        normalized_angles = []
        for angle in angles:
            while angle > 90:
                angle -= 180
            while angle < -90:
                angle += 180
            normalized_angles.append(angle)
        
        # Use median angle (more robust than mean)
        dominant_angle = np.median(normalized_angles)
        
        # Refine: if angle is very small, set to 0
        if abs(dominant_angle) < 0.5:
            dominant_angle = 0.0
        
        return (dominant_angle, is_vertical)
    
    def _detect_strip_vertical_edges_with_angle(
        self,
        image: np.ndarray,
        strip_region: Tuple[int, int, int, int],
        angle: float,
        is_vertical: bool
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Step 2: Detect top/bottom edges accounting for strip angle.
        Uses multiple methods: projection analysis, edge detection, and brightness analysis.
        
        Returns:
            (top_edge, bottom_edge) in original image coordinates
        """
        if not is_vertical:
            # For horizontal strips, swap logic
            # Top/bottom become left/right
            left_edge, right_edge = self._detect_strip_horizontal_edges_with_angle(
                image, strip_region[0], strip_region[1], 
                strip_region[1] + strip_region[3], angle, is_vertical
            )
            # For horizontal, return as top/bottom (but they're actually left/right)
            return (left_edge, right_edge)
        
        h, w = image.shape[:2]
        x_approx, y_approx, w_approx, h_approx = strip_region
        
        # IMPORTANT: Search WITHIN the general area, not the whole image
        # The general area gives us approximate bounds: y_approx to y_approx + h_approx
        # We need to find the actual top/bottom edges within this region
        
        # Crop to the general area (with small margin for edge detection)
        margin = 20  # Small margin for edge detection context
        y_start = max(0, y_approx - margin)
        y_end = min(h, y_approx + h_approx + margin)
        x_start = max(0, x_approx - margin)
        x_end = min(w, x_approx + w_approx + margin)
        
        # Extract the region to search
        search_roi = image[y_start:y_end, x_start:x_end]
        h_search, w_search = search_roi.shape[:2]
        
        # Rotate the search region if angle is significant (to align for better edge detection)
        y_offset_in_roi = 0
        if abs(angle) > 0.5:
            center = (w_search // 2, h_search // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Expand to avoid clipping during rotation
            expanded_size = int(max(w_search, h_search) * 1.5)
            search_roi = cv2.warpAffine(search_roi, rotation_matrix, (expanded_size, expanded_size))
            h_search, w_search = search_roi.shape[:2]
            # Calculate offset: how much padding was added
            y_offset_in_roi = (expanded_size - (y_end - y_start)) // 2
        
        gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Vertical projection (sum per row) - finds where strip is brightest
        vertical_projection = np.sum(gray_roi, axis=1)
        
        # Smooth with larger window
        window_size = max(10, int(h_search * 0.05))
        kernel = np.ones(window_size) / window_size
        smoothed_v = np.convolve(vertical_projection, kernel, mode='same')
        
        # Find peak (brightest region = strip center)
        peak_y_in_roi = np.argmax(smoothed_v)
        peak_value = smoothed_v[peak_y_in_roi]
        mean_value = np.mean(smoothed_v)
        
        # Use adaptive threshold based on peak vs mean
        threshold = mean_value + (peak_value - mean_value) * 0.3  # 30% above mean
        
        # Find top edge: scan from top of search region, find where brightness becomes consistently high
        # Constrain search to within reasonable bounds (not at very edges of ROI)
        search_start = max(5, int(h_search * 0.05))  # Start 5% into ROI
        search_end = min(h_search - 5, int(h_search * 0.95))  # End 5% before ROI edge
        
        top_edge_in_roi = search_start
        consecutive_high = 0
        required_consecutive = max(5, int(h_search * 0.01))
        
        for i in range(search_start, peak_y_in_roi):
            if smoothed_v[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    top_edge_in_roi = i - required_consecutive + 1
                    break
            else:
                consecutive_high = 0
        
        # Find bottom edge: scan from bottom of search region
        bottom_edge_in_roi = search_end
        consecutive_high = 0
        
        for i in range(search_end - 1, peak_y_in_roi, -1):
            if smoothed_v[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    bottom_edge_in_roi = i + required_consecutive - 1
                    break
            else:
                consecutive_high = 0
        
        # Method 2: Use edge detection as validation
        edges = cv2.Canny(gray_roi, 50, 150)
        # Find horizontal edges (top and bottom of strip)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_search // 2, 1))
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Find topmost and bottommost strong horizontal edges within search bounds
        edge_projection = np.sum(horizontal_edges, axis=1)
        edge_threshold = np.max(edge_projection) * 0.5
        
        top_edge_from_edges = search_start
        for i in range(search_start, h_search):
            if edge_projection[i] > edge_threshold:
                top_edge_from_edges = i
                break
        
        bottom_edge_from_edges = search_end
        for i in range(search_end - 1, -1, -1):
            if edge_projection[i] > edge_threshold:
                bottom_edge_from_edges = i
                break
        
        # Combine both methods (use projection as primary, edges as validation)
        top_edge_in_roi = min(top_edge_in_roi, top_edge_from_edges)
        bottom_edge_in_roi = max(bottom_edge_in_roi, bottom_edge_from_edges)
        
        # Convert back to original image coordinates
        if abs(angle) > 0.5:
            # Account for rotation and expansion
            top_edge = y_start + top_edge_in_roi - y_offset_in_roi
            bottom_edge = y_start + bottom_edge_in_roi - y_offset_in_roi
        else:
            top_edge = y_start + top_edge_in_roi
            bottom_edge = y_start + bottom_edge_in_roi
        
        # IMPORTANT: Constrain to general area bounds (with small tolerance)
        # The general area tells us where the strip approximately is
        # Top edge should be within the general area, bottom edge too
        tolerance = int(h * 0.03)  # 3% tolerance (smaller, more strict)
        
        # Top edge must be >= y_approx - tolerance and <= y_approx + h_approx
        # Bottom edge must be >= y_approx and <= y_approx + h_approx + tolerance
        top_edge = max(y_approx - tolerance, min(y_approx + h_approx, top_edge))
        bottom_edge = min(y_approx + h_approx + tolerance, max(y_approx, bottom_edge))
        
        # Ensure top < bottom
        if top_edge >= bottom_edge:
            # Fallback: use general area bounds
            top_edge = y_approx
            bottom_edge = y_approx + h_approx
        
        # Clamp to image bounds and ensure integers
        top_edge = int(max(0, min(h - 1, top_edge)))
        bottom_edge = int(max(0, min(h - 1, bottom_edge)))
        
        # Validate
        strip_height = bottom_edge - top_edge
        if strip_height < h * 0.15:  # At least 15% of image height
            return (None, None)
        
        return (top_edge, bottom_edge)
    
    def _detect_strip_horizontal_edges_with_angle(
        self,
        image: np.ndarray,
        x_approx: int,
        top_edge: int,
        bottom_edge: int,
        angle: float,
        is_vertical: bool
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Step 3: Detect left/right edges accounting for lighting artifacts and angle.
        Uses projection analysis and edge detection to find actual strip boundaries.
        
        Returns:
            (left_edge, right_edge) in original image coordinates
        """
        # Ensure all coordinates are Python integers
        x_approx = int(x_approx)
        top_edge = int(top_edge)
        bottom_edge = int(bottom_edge)
        
        h, w = image.shape[:2]
        strip_height = bottom_edge - top_edge
        
        # Extract wider region for analysis
        margin = int(w * 0.2)  # Increased margin
        x_start = int(max(0, x_approx - margin))
        x_end = int(min(w, x_approx + w * 0.3 + margin))  # Ensure we have enough width
        width_analysis = x_end - x_start
        
        # Crop to analysis region
        analysis_roi = image[top_edge:bottom_edge, x_start:x_end]
        h_roi, w_roi = analysis_roi.shape[:2]
        
        # If angle is significant, rotate for better edge detection
        x_offset = 0
        if abs(angle) > 0.5 and is_vertical:
            center = (w_roi // 2, h_roi // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Expand to avoid clipping
            expanded_size = int(max(w_roi, h_roi) * 1.5)
            analysis_roi = cv2.warpAffine(analysis_roi, rotation_matrix, (expanded_size, expanded_size))
            w_roi = expanded_size
            h_roi = expanded_size
            x_offset = (expanded_size - (x_end - x_start)) // 2
        
        gray_roi = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Horizontal projection (sum per column)
        horizontal_projection = np.sum(gray_roi, axis=0)
        
        # Smooth with larger window
        window_size = max(5, int(w_roi * 0.02))
        kernel = np.ones(window_size) / window_size
        smoothed_h = np.convolve(horizontal_projection, kernel, mode='same')
        
        # Find peak (center of strip)
        peak_x_roi = np.argmax(smoothed_h)
        peak_value = smoothed_h[peak_x_roi]
        mean_value = np.mean(smoothed_h)
        
        # Use adaptive threshold
        threshold = mean_value + (peak_value - mean_value) * 0.3  # 30% above mean
        
        # Find edges with consecutive high values (accounts for thin bright/dark lines)
        required_consecutive = max(5, int(w_roi * 0.01))
        
        # Left edge: scan from left, find where brightness becomes consistently high
        left_edge_roi = 0
        consecutive_high = 0
        for i in range(0, peak_x_roi):
            if smoothed_h[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    left_edge_roi = i - required_consecutive + 1
                    break
            else:
                consecutive_high = 0
        
        # Right edge: scan from right
        right_edge_roi = w_roi - 1
        consecutive_high = 0
        for i in range(w_roi - 1, peak_x_roi, -1):
            if smoothed_h[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    right_edge_roi = i + required_consecutive - 1
                    break
            else:
                consecutive_high = 0
        
        # Method 2: Use vertical edge detection as validation
        edges = cv2.Canny(gray_roi, 50, 150)
        # Find vertical edges (left and right of strip)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_roi // 2))
        vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
        
        # Find leftmost and rightmost strong vertical edges
        edge_projection = np.sum(vertical_edges, axis=0)
        edge_threshold = np.max(edge_projection) * 0.5
        
        left_edge_from_edges = 0
        for i in range(w_roi):
            if edge_projection[i] > edge_threshold:
                left_edge_from_edges = i
                break
        
        right_edge_from_edges = w_roi - 1
        for i in range(w_roi - 1, -1, -1):
            if edge_projection[i] > edge_threshold:
                right_edge_from_edges = i
                break
        
        # Combine both methods (use projection as primary, edges as validation)
        left_edge_roi = min(left_edge_roi, left_edge_from_edges)
        right_edge_roi = max(right_edge_roi, right_edge_from_edges)
        
        # Convert back to original coordinates and ensure integers
        if abs(angle) > 0.5:
            left_edge = int(x_start + left_edge_roi - x_offset)
            right_edge = int(x_start + right_edge_roi - x_offset)
        else:
            left_edge = int(x_start + left_edge_roi)
            right_edge = int(x_start + right_edge_roi)
        
        # Clamp to image bounds
        left_edge = max(0, min(w - 1, left_edge))
        right_edge = max(0, min(w - 1, right_edge))
        
        # Validate
        strip_width = right_edge - left_edge
        if strip_width < 50 or strip_width > w * 0.5:
            return (None, None)
        
        return (left_edge, right_edge)
    
    def _get_strip_corners(
        self,
        left: int,
        right: int,
        top: int,
        bottom: int,
        angle: float
    ) -> np.ndarray:
        """Get the four corners of the strip accounting for rotation."""
        # Calculate center
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        # Calculate dimensions
        width = right - left
        height = bottom - top
        
        # Get corners in local coordinates (relative to center)
        corners_local = np.array([
            [-width/2, -height/2],  # Top-left
            [width/2, -height/2],   # Top-right
            [width/2, height/2],     # Bottom-right
            [-width/2, height/2]    # Bottom-left
        ], dtype=np.float32)
        
        # Rotate corners
        if abs(angle) > 0.1:
            angle_rad = np.radians(angle)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corners_rotated = corners_local @ rotation_matrix.T
        else:
            corners_rotated = corners_local
        
        # Translate to image coordinates
        corners = corners_rotated + np.array([center_x, center_y])
        
        return corners.astype(np.int32)
    
    def _extract_and_align_strip(
        self,
        image: np.ndarray,
        corners: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract strip region and rotate it to be perfectly aligned.
        
        Returns:
            (aligned_strip_image, transform_matrix)
        """
        # Calculate dimensions of aligned strip
        width = int(np.linalg.norm(corners[1] - corners[0]))
        height = int(np.linalg.norm(corners[3] - corners[0]))
        
        # Define destination corners (aligned rectangle)
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Get perspective transform
        transform_matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            dst_corners
        )
        
        # Warp the image
        aligned_strip = cv2.warpPerspective(
            image,
            transform_matrix,
            (width, height)
        )
        
        return (aligned_strip, transform_matrix)
    
    def _transform_pad_coordinates_to_original(
        self,
        pad_regions: List[Tuple[int, int, int, int]],
        transform_matrix: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Transform pad coordinates from aligned strip back to original image."""
        # Get inverse transform
        inv_transform = np.linalg.inv(transform_matrix)
        
        transformed_regions = []
        for x, y, w, h in pad_regions:
            # Transform the four corners of the pad
            corners = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)
            
            # Add homogeneous coordinate
            corners_homogeneous = np.column_stack([corners, np.ones(4)])
            
            # Transform
            transformed_corners = (inv_transform @ corners_homogeneous.T).T
            transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2:3]
            
            # Get bounding box
            x_min = int(np.min(transformed_corners[:, 0]))
            y_min = int(np.min(transformed_corners[:, 1]))
            x_max = int(np.max(transformed_corners[:, 0]))
            y_max = int(np.max(transformed_corners[:, 1]))
            
            transformed_regions.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return transformed_regions
    
    def _detect_strip_orientation_and_vertical_edges(
        self, 
        image: np.ndarray, 
        strip_region: Tuple[int, int, int, int]
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Step 2: Determine orientation and detect top/bottom edges.
        
        Returns:
            (is_vertical, top_edge, bottom_edge)
            - is_vertical: True if strip is vertical, False if horizontal
            - top_edge: y-coordinate of top edge (for vertical) or None
            - bottom_edge: y-coordinate of bottom edge (for vertical) or None
        """
        h, w = image.shape[:2]
        x_approx, y_approx, w_approx, h_approx = strip_region
        
        # Crop to approximate strip region for analysis
        roi = image[y_approx:y_approx+h_approx, x_approx:x_approx+w_approx]
        h_roi, w_roi = roi.shape[:2]
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate vertical projection (sum per row) to find top/bottom
        vertical_projection = np.sum(gray_roi, axis=1)
        
        # Smooth the projection
        window_size = max(5, int(h_roi * 0.02))
        kernel = np.ones(window_size) / window_size
        smoothed_v = np.convolve(vertical_projection, kernel, mode='same')
        
        # Find peak (brightest region)
        peak_y = np.argmax(smoothed_v)
        peak_value = smoothed_v[peak_y]
        
        # Determine if vertical or horizontal by comparing dimensions
        # Vertical: height >> width, Horizontal: width >> height
        is_vertical = h_roi > w_roi * 1.5
        
        if not is_vertical:
            # Horizontal strip - not yet implemented
            return (False, None, None)
        
        # For vertical strip, find top and bottom edges
        # Top edge: where brightness drops significantly from top
        # Bottom edge: where brightness drops significantly from bottom
        
        threshold = peak_value * 0.7  # 70% of peak
        
        # Find top edge (scanning from top)
        top_edge_roi = 0
        for i in range(0, peak_y):
            if smoothed_v[i] >= threshold:
                top_edge_roi = i
            else:
                break
        
        # Find bottom edge (scanning from bottom)
        bottom_edge_roi = h_roi - 1
        for i in range(h_roi - 1, peak_y, -1):
            if smoothed_v[i] >= threshold:
                bottom_edge_roi = i
            else:
                break
        
        # Convert back to original image coordinates
        top_edge = y_approx + top_edge_roi
        bottom_edge = y_approx + bottom_edge_roi
        
        # Validate edges
        if bottom_edge - top_edge < h * 0.2:  # At least 20% of image height
            return (True, None, None)
        
        return (True, top_edge, bottom_edge)
    
    def _detect_strip_horizontal_edges(
        self,
        image: np.ndarray,
        x_approx: int,
        top_edge: int,
        bottom_edge: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Step 3: Detect left/right edges accounting for lighting artifacts.
        Lighting can create thin bright lines on one edge and thin dark lines on the other.
        
        Returns:
            (left_edge, right_edge) in original image coordinates
        """
        h, w = image.shape[:2]
        strip_height = bottom_edge - top_edge
        
        # Extract a wider region for analysis (include some margin)
        margin = int(w * 0.1)  # 10% margin on each side
        x_start = max(0, x_approx - margin)
        x_end = min(w, x_approx + x_approx + margin)
        width_analysis = x_end - x_start
        
        # Crop to analysis region
        analysis_roi = image[top_edge:bottom_edge, x_start:x_end]
        gray_roi = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal projection (sum per column)
        horizontal_projection = np.sum(gray_roi, axis=0)
        
        # Smooth to reduce noise
        window_size = max(3, int(width_analysis * 0.01))
        kernel = np.ones(window_size) / window_size
        smoothed_h = np.convolve(horizontal_projection, kernel, mode='same')
        
        # Find peak (center of strip)
        peak_x_roi = np.argmax(smoothed_h)
        peak_value = smoothed_h[peak_x_roi]
        
        # Find edges using a more sophisticated approach
        # Account for thin bright/dark lines at edges
        
        # Method: Find where brightness stabilizes (not just threshold)
        # Look for regions where brightness is consistently high
        
        threshold = peak_value * 0.8  # 80% of peak
        
        # Find left edge: scan from left, find where brightness becomes consistently high
        left_edge_roi = 0
        consecutive_high = 0
        required_consecutive = max(3, int(width_analysis * 0.01))  # Need several consecutive high values
        
        for i in range(0, peak_x_roi):
            if smoothed_h[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    left_edge_roi = i - required_consecutive + 1
                    break
            else:
                consecutive_high = 0
        
        # Find right edge: scan from right
        right_edge_roi = width_analysis - 1
        consecutive_high = 0
        
        for i in range(width_analysis - 1, peak_x_roi, -1):
            if smoothed_h[i] >= threshold:
                consecutive_high += 1
                if consecutive_high >= required_consecutive:
                    right_edge_roi = i + required_consecutive - 1
                    break
            else:
                consecutive_high = 0
        
        # Convert to original image coordinates
        left_edge = x_start + left_edge_roi
        right_edge = x_start + right_edge_roi
        
        # Validate edges
        strip_width = right_edge - left_edge
        if strip_width < 50 or strip_width > w * 0.5:
            return (None, None)
        
        return (left_edge, right_edge)
    
    def _detect_colored_squares_in_strip(
        self,
        strip_image: np.ndarray,
        expected_count: int,
        strip_width: int,
        strip_height: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Step 4: Detect colored squares within validated strip.
        Uses known strip dimensions to estimate square size and spacing.
        Accounts for pattern: handle + white squares alternating with colored squares.
        
        Returns:
            List of (x, y, width, height) tuples for each colored square
        """
        h_strip, w_strip = strip_image.shape[:2]
        
        # Estimate square size based on strip width
        # Squares are roughly square, slightly smaller than strip width
        estimated_square_size = int(w_strip * 0.85)  # 85% of strip width
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(strip_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for colored regions (not white/very light)
        # Use more lenient thresholds for "almost white" colors
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])  # Higher saturation tolerance
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_colored = cv2.bitwise_not(mask_white)
        
        # Apply morphological operations
        kernel_size = max(3, estimated_square_size // 10)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_CLOSE, kernel)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        min_area = (estimated_square_size * estimated_square_size) * 0.3  # At least 30% of estimated square
        max_area = (estimated_square_size * estimated_square_size) * 2.0   # At most 2x estimated square
        
        square_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, rw, rh = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (should be roughly square)
                aspect_ratio = rw / rh if rh > 0 else 0
                if 0.6 < aspect_ratio < 1.5:  # Roughly square
                    # Check if size is close to estimated
                    size_ratio = (rw + rh) / 2 / estimated_square_size
                    if 0.5 < size_ratio < 1.5:  # Within 50% of estimated size
                        square_candidates.append((x, y, rw, rh))
        
        # Sort by y-coordinate (top to bottom)
        square_candidates.sort(key=lambda s: s[1])
        
        # If we found more than expected, filter by spacing
        if len(square_candidates) > expected_count:
            # Estimate spacing between squares
            if len(square_candidates) >= 2:
                avg_spacing = np.mean([
                    square_candidates[i+1][1] - square_candidates[i][1] 
                    for i in range(len(square_candidates) - 1)
                ])
                
                # Keep squares that are roughly evenly spaced
                filtered = [square_candidates[0]]  # Keep first
                for i in range(1, len(square_candidates)):
                    spacing = square_candidates[i][1] - filtered[-1][1]
                    if 0.7 * avg_spacing < spacing < 1.3 * avg_spacing:
                        filtered.append(square_candidates[i])
                        if len(filtered) >= expected_count:
                            break
                
                square_candidates = filtered[:expected_count]
            else:
                square_candidates = square_candidates[:expected_count]
        
        return square_candidates
    
    def _detect_test_strip(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the white/light test strip region in the image.
        Uses multiple methods: projection-based (most accurate), color detection, edge detection.
        
        Args:
            image: OpenCV image array in BGR format
        
        Returns:
            (x, y, width, height) of test strip region, or None if not found
        """
        h, w = image.shape[:2]
        is_portrait = h > w
        
        # Method 1: Projection-based (most accurate for width)
        strip_region = self._detect_strip_by_projection(image, is_portrait)
        if strip_region:
            return strip_region
        
        # Method 2: Detect light/white regions
        strip_region = self._detect_strip_by_color(image, is_portrait)
        if strip_region:
            return strip_region
        
        # Method 3: Detect by edges (find rectangular boundaries)
        strip_region = self._detect_strip_by_edges(image, is_portrait)
        if strip_region:
            return strip_region
        
        # Method 4: Use center region assumption (fallback)
        # For portrait images, strip is likely in center horizontally
        if is_portrait:
            center_x = w // 2
            strip_width = int(w * 0.2)  # Assume strip is 20% of width (tighter)
            x = center_x - strip_width // 2
            y = int(h * 0.1)  # Start 10% from top
            rh = int(h * 0.8)  # 80% of height
            return (x, y, strip_width, rh)
        
        return None
    
    def _detect_strip_by_projection(
        self, 
        image: np.ndarray, 
        is_portrait: bool
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect strip using horizontal projection to find the brightest/most uniform region.
        This method is more accurate for finding strip width.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal projection (sum of pixel values per column)
        projection = np.sum(gray, axis=0)
        
        # Smooth the projection to reduce noise
        # Use a simple moving average
        window_size = max(10, int(w * 0.02))  # 2% of width
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find the peak (brightest region = strip)
        peak_idx = np.argmax(smoothed)
        peak_value = smoothed[peak_idx]
        
        # Find width around peak (where values are above threshold)
        # Use a threshold that's relative to the peak
        threshold = peak_value * 0.85  # 85% of peak value
        
        # Find left and right boundaries
        left_bound = peak_idx
        right_bound = peak_idx
        
        # Expand left
        for i in range(peak_idx, -1, -1):
            if smoothed[i] >= threshold:
                left_bound = i
            else:
                break
        
        # Expand right
        for i in range(peak_idx, len(smoothed)):
            if smoothed[i] >= threshold:
                right_bound = i
            else:
                break
        
        strip_width = right_bound - left_bound
        
        # Validate width (should be reasonable for a test strip)
        min_width = int(w * 0.05)   # At least 5% of image width
        max_width = int(w * 0.4)    # At most 40% of image width
        
        if min_width <= strip_width <= max_width:
            # For portrait, strip runs vertically
            x = left_bound
            y = int(h * 0.05)  # Start 5% from top
            rh = int(h * 0.9)  # 90% of height
            
            # Add small padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            strip_width = min(w - x, strip_width + 2 * padding)
            rh = min(h - y, rh + 2 * padding)
            
            return (x, y, strip_width, rh)
        
        return None
    
    def _detect_strip_by_color(
        self, 
        image: np.ndarray, 
        is_portrait: bool
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect strip using light/white color detection."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # More lenient white/light detection
        # Lower value threshold, higher saturation tolerance
        lower_light = np.array([0, 0, 150])  # Lower value threshold
        upper_light = np.array([180, 60, 255])  # Higher saturation tolerance
        mask_light = cv2.inRange(hsv, lower_light, upper_light)
        
        # Apply morphological operations to connect nearby regions
        # Use larger kernel for portrait images (tall strip)
        kernel_size = 30 if is_portrait else 20
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
        mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find candidates based on aspect ratio and size
        strip_candidates = []
        min_area = (w * h) * 0.01  # Much lower: 1% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, rw, rh = cv2.boundingRect(contour)
            
            # Check aspect ratio
            if is_portrait:
                aspect_ratio = rh / rw if rw > 0 else 0
                # Portrait: strip should be tall (aspect ratio > 2.0)
                if aspect_ratio > 2.0:
                    strip_candidates.append((x, y, rw, rh, area))
            else:
                aspect_ratio = rw / rh if rh > 0 else 0
                # Landscape: strip should be wide (aspect ratio > 2.0)
                if aspect_ratio > 2.0:
                    strip_candidates.append((x, y, rw, rh, area))
        
        if strip_candidates:
            # Sort by area and take largest
            strip_candidates.sort(key=lambda c: c[4], reverse=True)
            x, y, rw, rh, _ = strip_candidates[0]
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            rw = min(w - x, rw + 2 * padding)
            rh = min(h - y, rh + 2 * padding)
            
            return (x, y, rw, rh)
        
        return None
    
    def _detect_strip_by_edges(
        self, 
        image: np.ndarray, 
        is_portrait: bool
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect strip using edge detection to find rectangular boundaries."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((10, 10), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find rectangular regions with appropriate aspect ratio
        strip_candidates = []
        min_area = (w * h) * 0.05  # 5% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, rw, rh = cv2.boundingRect(contour)
                
                # Check aspect ratio
                if is_portrait:
                    aspect_ratio = rh / rw if rw > 0 else 0
                    if aspect_ratio > 2.0:
                        strip_candidates.append((x, y, rw, rh, area))
                else:
                    aspect_ratio = rw / rh if rh > 0 else 0
                    if aspect_ratio > 2.0:
                        strip_candidates.append((x, y, rw, rh, area))
        
        if strip_candidates:
            # Sort by area and take largest
            strip_candidates.sort(key=lambda c: c[4], reverse=True)
            x, y, rw, rh, _ = strip_candidates[0]
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            rw = min(w - x, rw + 2 * padding)
            rh = min(h - y, rh + 2 * padding)
            
            return (x, y, rw, rh)
        
        return None
    
    def _detect_pads_in_strip(
        self, 
        strip_image: np.ndarray, 
        expected_count: int,
        is_portrait: bool
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect colored pads within the test strip region.
        
        Args:
            strip_image: Cropped image of just the test strip
            expected_count: Expected number of pads
            is_portrait: Whether original image was portrait orientation
        
        Returns:
            List of (x, y, width, height, area) tuples for pad candidates
        """
        h_strip, w_strip = strip_image.shape[:2]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(strip_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for colored regions (not white)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_colored = cv2.bitwise_not(mask_white)
        
        # Apply morphological operations to clean up and connect pad regions
        kernel_size = max(3, min(w_strip, h_strip) // 50)  # Adaptive kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_CLOSE, kernel)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_OPEN, kernel)
        
        # Find contours of colored regions
        contours, _ = cv2.findContours(mask_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        # Pads should be a reasonable size relative to strip
        min_area = (w_strip * h_strip) * 0.01  # 1% of strip area
        max_area = (w_strip * h_strip) * 0.3   # 30% of strip area
        
        pad_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, rw, rh = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (pads are roughly square to slightly rectangular)
                aspect_ratio = rw / rh if rh > 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    # Check if region is reasonably compact
                    extent = area / (rw * rh) if (rw * rh) > 0 else 0
                    if extent > 0.5:  # At least 50% of bounding box is filled
                        pad_candidates.append((x, y, rw, rh, area))
        
        # If we found too many, filter by size and position
        if len(pad_candidates) > expected_count:
            # Sort by area and take largest
            pad_candidates.sort(key=lambda p: p[4], reverse=True)
            pad_candidates = pad_candidates[:expected_count * 2]  # Take top 2x for filtering
            
            # If portrait, pads should be arranged vertically (sort by y)
            # If landscape, pads should be arranged horizontally (sort by x)
            if is_portrait:
                pad_candidates.sort(key=lambda p: p[1])  # Sort by y
            else:
                pad_candidates.sort(key=lambda p: p[0])  # Sort by x
            
            # Take evenly spaced candidates
            if len(pad_candidates) > expected_count:
                step = len(pad_candidates) / expected_count
                selected = []
                for i in range(expected_count):
                    idx = int(i * step)
                    if idx < len(pad_candidates):
                        selected.append(pad_candidates[idx])
                pad_candidates = selected
        
        return pad_candidates
    
    def _detect_pads_by_color(
        self, 
        image: np.ndarray, 
        expected_count: int
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect pads using color-based segmentation.
        Finds colored regions (non-white) on white background.
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for colored regions (not white)
        # White in HSV: low saturation, high value
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_colored = cv2.bitwise_not(mask_white)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_CLOSE, kernel)
        mask_colored = cv2.morphologyEx(mask_colored, cv2.MORPH_OPEN, kernel)
        
        # Find contours of colored regions
        contours, _ = cv2.findContours(mask_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        # For portrait images, pads are smaller relative to image size
        min_area = (w * h) * 0.0005  # 0.05% of image (much smaller threshold)
        max_area = (w * h) * 0.1    # 10% of image
        
        pad_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, rw, rh = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (pads are roughly square to slightly rectangular)
                aspect_ratio = rw / rh if rh > 0 else 0
                if 0.3 < aspect_ratio < 3.0:  # More lenient aspect ratio
                    # Check if region is reasonably compact (not too elongated)
                    extent = area / (rw * rh) if (rw * rh) > 0 else 0
                    if extent > 0.5:  # At least 50% of bounding box is filled
                        pad_candidates.append((x, y, rw, rh, area))
        
        return pad_candidates
    
    def _detect_pads_by_edges(
        self, 
        image: np.ndarray, 
        expected_count: int
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect pads using edge detection (fallback method).
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Otsu's threshold for better results
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        min_area = (w * h) * 0.0005  # 0.05% of image
        max_area = (w * h) * 0.1     # 10% of image
        
        pad_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, rw, rh = cv2.boundingRect(contour)
                aspect_ratio = rw / rh if rh > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    extent = area / (rw * rh) if (rw * rh) > 0 else 0
                    if extent > 0.4:
                        pad_candidates.append((x, y, rw, rh, area))
        
        return pad_candidates
    
    def _merge_nearby_pads(
        self, 
        candidates: List[Tuple], 
        target_count: int
    ) -> List[Tuple]:
        """Merge nearby pad candidates."""
        if len(candidates) <= target_count:
            return candidates
        
        # Simple approach: take evenly spaced candidates
        step = len(candidates) / target_count
        selected = []
        for i in range(target_count):
            idx = int(i * step)
            selected.append(candidates[idx])
        
        return selected
    
    def _detect_pads_alternative(
        self, 
        image: np.ndarray, 
        expected_count: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Alternative pad detection using horizontal projection.
        Assumes pads are arranged horizontally.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate horizontal projection (sum of pixel values per column)
        projection = np.sum(gray, axis=0)
        
        # Find peaks in projection (these indicate pad locations)
        # Use a simple peak detection
        threshold = np.mean(projection) * 0.8
        peaks = []
        for i in range(1, len(projection) - 1):
            if projection[i] > threshold and projection[i] > projection[i-1] and projection[i] > projection[i+1]:
                peaks.append(i)
        
        # Estimate pad width
        if len(peaks) >= expected_count:
            # Use evenly spaced regions
            regions = []
            pad_width = w // (expected_count + 1)
            pad_height = int(h * 0.6)  # Assume pads take up 60% of height
            pad_y = int(h * 0.2)  # Start 20% from top
            
            for i in range(expected_count):
                x = int((i + 0.5) * pad_width)
                regions.append((x - pad_width//2, pad_y, pad_width, pad_height))
            
            return [(x, y, w, h, w*h) for x, y, w, h in regions]
        
        return []
    
    def _get_inner_region(
        self,
        region: Tuple[int, int, int, int],
        edge_exclusion: float = 0.15
    ) -> Tuple[int, int, int, int]:
        """
        Calculate inner region excluding edges.
        
        Args:
            region: (x, y, width, height) outer region
            edge_exclusion: Fraction of region to exclude from each edge (default: 15%)
        
        Returns:
            (x, y, width, height) inner region
        """
        x, y, w, h = region
        # Calculate exclusion amounts
        x_exclude = int(w * edge_exclusion)
        y_exclude = int(h * edge_exclusion)
        
        # Inner region
        inner_x = x + x_exclude
        inner_y = y + y_exclude
        inner_w = max(1, w - 2 * x_exclude)
        inner_h = max(1, h - 2 * y_exclude)
        
        return (inner_x, inner_y, inner_w, inner_h)
    
    def _detect_white_regions_between_pads(
        self,
        lab_image: np.ndarray,
        pad_regions: List[PadRegion],
        bgr_image: np.ndarray,
        edge_padding: int = 5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect white regions between pads for use as white reference.
        
        Args:
            lab_image: Image in LAB color space
            pad_regions: List of detected pad regions
            bgr_image: Original BGR image for visualization
            edge_padding: Pixels to exclude from pad edges (default: 5px)
        
        Returns:
            List of (x, y, width, height) white regions
        """
        if len(pad_regions) < 2:
            return []
        
        white_regions = []
        h, w = lab_image.shape[:2]
        
        # Sort pads by position (top to bottom for vertical strips)
        sorted_pads = sorted(pad_regions, key=lambda p: (p['y'], p['x']))
        
        # Check if vertical or horizontal strip
        y_range = sorted_pads[-1]['y'] + sorted_pads[-1]['height'] - sorted_pads[0]['y']
        x_range = max(p['x'] + p['width'] for p in sorted_pads) - min(p['x'] for p in sorted_pads)
        is_vertical = y_range > x_range
        
        for i in range(len(sorted_pads) - 1):
            pad1 = sorted_pads[i]
            pad2 = sorted_pads[i + 1]
            
            if is_vertical:
                # Vertical strip: white region is between pads vertically
                # Use horizontal center of pads, vertical space between them
                center_x = (pad1['x'] + pad1['width'] // 2 + pad2['x'] + pad2['width'] // 2) // 2
                pad1_bottom = pad1['y'] + pad1['height']
                pad2_top = pad2['y']
                
                # Region between pads, with padding away from pad edges
                gap_height = pad2_top - pad1_bottom
                if gap_height > (edge_padding * 2 + 5):  # Need space for padding + minimum region
                    # Start after padding from pad1 bottom, end before padding to pad2 top
                    region_y = pad1_bottom + edge_padding
                    region_h = gap_height - (edge_padding * 2)
                    
                    # Use narrower width, centered, with padding from pad edges
                    region_width = min(pad1['width'], pad2['width'], 40)  # Reduced from 50px
                    region_x = max(edge_padding, center_x - region_width // 2)
                    # Ensure we don't go beyond pad edges (add padding from left/right edges of pads)
                    pad1_left = pad1['x']
                    pad1_right = pad1['x'] + pad1['width']
                    pad2_left = pad2['x']
                    pad2_right = pad2['x'] + pad2['width']
                    
                    # Use the inner region between pads (avoid pad edges)
                    inner_left = max(pad1_left, pad2_left) + edge_padding
                    inner_right = min(pad1_right, pad2_right) - edge_padding
                    region_x = max(inner_left, center_x - region_width // 2)
                    region_w = min(w - region_x, inner_right - region_x, region_width)
                    
                    if region_w > 5 and region_h > 5:
                        # Check if region is actually white (high L, low a, low b)
                        roi = lab_image[region_y:region_y + region_h, region_x:region_x + region_w]
                        if roi.size > 0:
                            mean_lab = np.mean(roi.reshape(-1, 3), axis=0)
                            
                            # White criteria: L > 80, |a| < 20, |b| < 20
                            if mean_lab[0] > 80 and abs(mean_lab[1] - 128) < 20 and abs(mean_lab[2] - 128) < 20:
                                white_regions.append((region_x, region_y, region_w, region_h))
            else:
                # Horizontal strip: white region is between pads horizontally
                pad1_right = pad1['x'] + pad1['width']
                pad2_left = pad2['x']
                
                gap_width = pad2_left - pad1_right
                if gap_width > (edge_padding * 2 + 5):
                    # Start after padding from pad1 right, end before padding to pad2 left
                    region_x = pad1_right + edge_padding
                    region_w = gap_width - (edge_padding * 2)
                    
                    # Use narrower height, centered, with padding from pad edges
                    region_height = min(pad1['height'], pad2['height'], 40)  # Reduced from 50px
                    center_y = (pad1['y'] + pad1['height'] // 2 + pad2['y'] + pad2['height'] // 2) // 2
                    
                    # Ensure we don't go beyond pad edges (add padding from top/bottom edges of pads)
                    pad1_top = pad1['y']
                    pad1_bottom = pad1['y'] + pad1['height']
                    pad2_top = pad2['y']
                    pad2_bottom = pad2['y'] + pad2['height']
                    
                    # Use the inner region between pads (avoid pad edges)
                    inner_top = max(pad1_top, pad2_top) + edge_padding
                    inner_bottom = min(pad1_bottom, pad2_bottom) - edge_padding
                    region_y = max(inner_top, center_y - region_height // 2)
                    region_h = min(h - region_y, inner_bottom - region_y, region_height)
                    
                    if region_w > 5 and region_h > 5:
                        roi = lab_image[region_y:region_y + region_h, region_x:region_x + region_w]
                        if roi.size > 0:
                            mean_lab = np.mean(roi.reshape(-1, 3), axis=0)
                            
                            if mean_lab[0] > 80 and abs(mean_lab[1] - 128) < 20 and abs(mean_lab[2] - 128) < 20:
                                white_regions.append((region_x, region_y, region_w, region_h))
        
        return white_regions
    
    def _normalize_white_balance_with_regions(
        self,
        lab_image: np.ndarray,
        white_regions: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Normalize white balance using detected white regions between pads.
        
        Args:
            lab_image: Image in LAB color space
            white_regions: List of (x, y, width, height) white reference regions
        
        Returns:
            White-balanced image in LAB color space
        """
        if not white_regions:
            return lab_image
        
        # Collect all white reference pixels
        white_pixels = []
        for x, y, w, h in white_regions:
            roi = lab_image[y:y+h, x:x+w]
            white_pixels.append(roi.reshape(-1, 3))
        
        # Calculate average white reference LAB
        all_white = np.concatenate(white_pixels, axis=0)
        ref_lab = np.mean(all_white, axis=0)
        
        # Target white in OpenCV 8-bit LAB format:
        # L=255 (max brightness), a=128 (neutral), b=128 (neutral)
        target_lab = np.array([255.0, 128.0, 128.0])
        
        # Calculate adjustment factors
        adjustment = target_lab - ref_lab
        
        # Apply adjustment to entire image
        normalized = lab_image.astype(np.float32) + adjustment
        # Clamp all channels to valid 8-bit range (0-255)
        normalized[:, :, 0] = np.clip(normalized[:, :, 0], 0, 255)
        normalized[:, :, 1] = np.clip(normalized[:, :, 1], 0, 255)
        normalized[:, :, 2] = np.clip(normalized[:, :, 2], 0, 255)
        normalized = normalized.astype(np.uint8)
        
        self.logger.debug(f'White balance normalized using {len(white_regions)} regions: ref={ref_lab}, adjustment={adjustment}')
        return normalized
    
    def _extract_pad_color(
        self,
        lab_image: np.ndarray,
        region: Tuple[int, int, int, int],
        pad_index: int,
        quality_metrics: Dict,
        white_norm_success: bool,
        original_region: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Extract color from a single pad region with confidence calculation.
        
        Args:
            lab_image: Image in LAB color space
            region: (x, y, width, height) region
            pad_index: Index of the pad (0-based)
            quality_metrics: Image quality metrics dict
            white_norm_success: Whether white normalization succeeded
        
        Returns:
            Dictionary with pad color data and confidence scores
        """
        x, y, w, h = region
        
        # Extract LAB values with variance
        color_data = extract_lab_values_with_variance(lab_image, region)
        
        # Calculate confidence scores
        pad_detection_confidence = self._calculate_pad_detection_confidence(
            color_data,
            region,
            quality_metrics,
            white_norm_success
        )
        
        # Use original region for coordinates (outer bounding box)
        # but inner region was used for color extraction
        if original_region:
            x, y, w, h = original_region
        else:
            x, y, w, h = region
        
        # Create PadRegion with coordinates (outer region)
        pad_region = PadRegion(
            pad_index=pad_index,
            x=x,
            y=y,
            width=w,
            height=h,
            left=x,
            top=y,
            right=x + w,
            bottom=y + h
        )
        
        return {
            'pad_index': pad_index,
            'region': pad_region,  # Include coordinates
            'lab': {
                'L': color_data['L'],
                'a': color_data['a'],
                'b': color_data['b']
            },
            'confidence': float(pad_detection_confidence),
            'pad_detection_confidence': float(pad_detection_confidence),
            'color_variance': float(color_data['color_variance'])
        }
    
    def _calculate_pad_detection_confidence(
        self,
        color_data: Dict,
        region: Tuple[int, int, int, int],
        quality_metrics: Dict,
        white_norm_success: bool
    ) -> float:
        """
        Calculate pad detection confidence using weighted factors.
        
        Primary factors (70% weight):
        - Color variance within pad (lower = higher confidence)
        - Detection quality (found expected pads, clear boundaries)
        - Image quality (brightness, contrast, focus)
        
        Secondary factors (30% weight):
        - Pad characteristics (size, edge clarity)
        - White normalization success
        - Color extraction quality
        
        Args:
            color_data: Color data dict with variance info
            region: Pad region (x, y, width, height)
            quality_metrics: Image quality metrics
            white_norm_success: Whether white normalization succeeded
        
        Returns:
            Confidence score (0-1)
        """
        # Primary factors (70% weight)
        primary_weight = 0.7
        
        # 1. Color variance (lower variance = higher confidence)
        # Normalize variance: typical values 0-50, lower is better
        color_variance = color_data.get('color_variance', 0)
        variance_score = max(0, 1.0 - (color_variance / 50.0))
        variance_score = min(1.0, variance_score)
        
        # 2. Detection quality (assume good if we got here)
        detection_score = 1.0  # We detected the pad, so it's at least reasonable
        
        # 3. Image quality (average of brightness, contrast, focus)
        brightness = quality_metrics.get('brightness', 0.5)
        contrast = quality_metrics.get('contrast', 0.5)
        focus_score = quality_metrics.get('focus_score', 0.5)
        
        # Normalize each to 0-1 (they're already normalized)
        quality_score = (brightness + contrast + focus_score) / 3.0
        
        # Average primary factors
        primary_score = (variance_score + detection_score + quality_score) / 3.0
        
        # Secondary factors (30% weight)
        secondary_weight = 0.3
        
        # 1. Pad characteristics (size, aspect ratio)
        x, y, w, h = region
        aspect_ratio = w / h if h > 0 else 1.0
        # Ideal aspect ratio is around 1.5-2.0 for test strip pads
        ideal_ratio = 1.75
        ratio_score = 1.0 - min(1.0, abs(aspect_ratio - ideal_ratio) / ideal_ratio)
        
        # Size score (larger pads are generally better, but not too large)
        area = w * h
        # Assume ideal area is around 5% of image (will be normalized)
        size_score = min(1.0, area / (1000 * 100))  # Rough normalization
        
        pad_char_score = (ratio_score + size_score) / 2.0
        
        # 2. White normalization success
        white_norm_score = 1.0 if white_norm_success else 0.7
        
        # 3. Color extraction quality (based on variance)
        extraction_score = variance_score  # Reuse variance score
        
        # Average secondary factors
        secondary_score = (pad_char_score + white_norm_score + extraction_score) / 3.0
        
        # Weighted combination
        confidence = (primary_weight * primary_score) + (secondary_weight * secondary_score)
        
        # Ensure 0-1 range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence

