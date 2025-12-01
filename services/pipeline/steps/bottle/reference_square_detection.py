"""
Swatch Detection Service for Bottle Pipeline.

Pure OpenCV-based detection of color swatches on bottle labels.
NO OpenAI calls - fast and free.

Detects colored rectangles, clusters them into rows, and extracts LAB colors.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

from utils.color_conversion import bgr_to_lab, extract_lab_values, rgb_to_hex

logger = logging.getLogger(__name__)

# Validation limits
MAX_ROWS = 10  # Maximum number of rows (pads)
MAX_SWATCHES_PER_ROW = 12  # Maximum swatches per row
MAX_TOTAL_SWATCHES = 100  # Maximum total swatches


class SwatchDetectionService:
    """
    Service for detecting color swatches using pure OpenCV.
    
    No AI/ML required - uses contour detection and clustering.
    """
    
    def __init__(
        self,
        min_swatch_size: int = 15,
        max_swatch_size: int = 150,
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),
        row_clustering_threshold: int = 30
    ):
        """
        Initialize swatch detection service.
        
        Args:
            min_swatch_size: Minimum swatch dimension in pixels
            max_swatch_size: Maximum swatch dimension in pixels
            aspect_ratio_range: (min, max) aspect ratio for valid swatches
            row_clustering_threshold: Max Y-distance to consider same row
        """
        self.logger = logging.getLogger(__name__)
        self.min_swatch_size = min_swatch_size
        self.max_swatch_size = max_swatch_size
        self.aspect_ratio_range = aspect_ratio_range
        self.row_clustering_threshold = row_clustering_threshold
    
    def detect_swatches(
        self,
        image: np.ndarray,
        options: Optional[Dict] = None
    ) -> Dict:
        """
        Detect color swatches in image.
        
        Args:
            image: Input image (BGR format)
            options: Optional detection parameters:
                - min_swatch_size: int
                - max_swatch_size: int
                - expected_rows: int (hint for clustering)
        
        Returns:
            Detection result:
            {
                'success': bool,
                'data': {
                    'rows': [
                        {
                            'row_index': int,
                            'y_center': int,
                            'swatches': [
                                {
                                    'region': {'x', 'y', 'width', 'height'},
                                    'color': {'lab': {...}, 'hex': str}
                                }
                            ]
                        }
                    ],
                    'total_swatches': int,
                    'image_dimensions': {'width', 'height'},
                    'processing_time_ms': int
                }
            }
        """
        import time
        start_time = time.time()
        
        options = options or {}
        
        # Override defaults with options
        min_size = options.get('min_swatch_size', self.min_swatch_size)
        max_size = options.get('max_swatch_size', self.max_swatch_size)
        expected_rows = options.get('expected_rows')
        
        h, w = image.shape[:2]
        self.logger.info(f"Detecting swatches in {w}x{h} image")
        
        try:
            # Step 1: Find candidate rectangles
            candidates = self._find_rectangle_candidates(image, min_size, max_size)
            self.logger.info(f"Found {len(candidates)} candidate rectangles")
            
            if not candidates:
                return self._success_result([], w, h, start_time)
            
            # Step 2: Filter by color variance (swatches should be relatively uniform)
            filtered = self._filter_by_color_uniformity(image, candidates)
            self.logger.info(f"After uniformity filter: {len(filtered)} candidates")
            
            if not filtered:
                return self._success_result([], w, h, start_time)
            
            # Step 3: Cluster into rows
            rows = self._cluster_into_rows(filtered, expected_rows)
            self.logger.info(f"Clustered into {len(rows)} rows")
            
            # Step 4: Validate counts before proceeding
            total_candidates = sum(len(row['swatches']) for row in rows)
            
            if len(rows) > MAX_ROWS:
                error_msg = f"Detection error: found {len(rows)} rows, maximum is {MAX_ROWS}. Image may not be a valid bottle label."
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 'TOO_MANY_ROWS'
                }
            
            if total_candidates > MAX_TOTAL_SWATCHES:
                error_msg = f"Detection error: found {total_candidates} swatches, maximum is {MAX_TOTAL_SWATCHES}. Image may not be a valid bottle label."
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': 'TOO_MANY_SWATCHES'
                }
            
            for i, row in enumerate(rows):
                if len(row['swatches']) > MAX_SWATCHES_PER_ROW:
                    error_msg = f"Detection error: row {i} has {len(row['swatches'])} swatches, maximum is {MAX_SWATCHES_PER_ROW}"
                    self.logger.error(error_msg)
                    return {
                        'success': False,
                        'error': error_msg,
                        'error_code': 'TOO_MANY_SWATCHES_IN_ROW'
                    }
            
            # Step 5: Extract colors for each swatch
            lab_image = bgr_to_lab(image)
            rows_with_colors = self._extract_colors(image, lab_image, rows)
            
            # Count total swatches
            total_swatches = sum(len(row['swatches']) for row in rows_with_colors)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                'success': True,
                'data': {
                    'rows': rows_with_colors,
                    'total_swatches': total_swatches,
                    'image_dimensions': {'width': w, 'height': h},
                    'processing_time_ms': processing_time_ms
                }
            }
            
        except Exception as e:
            self.logger.error(f"Swatch detection error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'error_code': 'DETECTION_ERROR'
            }
    
    def _find_rectangle_candidates(
        self,
        image: np.ndarray,
        min_size: int,
        max_size: int
    ) -> List[Dict]:
        """
        Find candidate rectangular regions using multiple methods.
        
        Args:
            image: Input image (BGR)
            min_size: Minimum dimension
            max_size: Maximum dimension
        
        Returns:
            List of candidate regions with x, y, width, height
        """
        candidates = []
        
        # Method 1: Edge detection + contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Canny edge detection
        edges = cv2.Canny(filtered, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < min_size or h < min_size:
                continue
            if w > max_size or h > max_size:
                continue
            
            # Filter by aspect ratio
            aspect = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect <= self.aspect_ratio_range[1]):
                continue
            
            # Check if roughly rectangular (4-6 corners after approximation)
            if 4 <= len(approx) <= 8:
                candidates.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'aspect_ratio': aspect
                })
        
        # Method 2: Adaptive thresholding for high-contrast swatches
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours2:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < min_size or h < min_size:
                continue
            if w > max_size or h > max_size:
                continue
            
            aspect = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect <= self.aspect_ratio_range[1]):
                continue
            
            # Check if this overlaps with existing candidate
            is_duplicate = False
            for existing in candidates:
                iou = self._calculate_iou(
                    (x, y, w, h),
                    (existing['x'], existing['y'], existing['width'], existing['height'])
                )
                if iou > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                candidates.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'aspect_ratio': aspect
                })
        
        return candidates
    
    def _calculate_iou(
        self,
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union between two rectangles."""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if right <= left or bottom <= top:
            return 0.0
        
        intersection = (right - left) * (bottom - top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_by_color_uniformity(
        self,
        image: np.ndarray,
        candidates: List[Dict],
        max_std_threshold: float = 50.0
    ) -> List[Dict]:
        """
        Filter candidates by color uniformity.
        
        Swatches should have relatively uniform color (low standard deviation).
        
        Args:
            image: Input image (BGR)
            candidates: List of candidate regions
            max_std_threshold: Maximum allowed color std deviation
        
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for candidate in candidates:
            x, y, w, h = candidate['x'], candidate['y'], candidate['width'], candidate['height']
            
            # Use center region to avoid edges
            margin = max(2, min(w, h) // 6)
            cx = x + margin
            cy = y + margin
            cw = max(1, w - 2 * margin)
            ch = max(1, h - 2 * margin)
            
            # Extract region
            roi = image[cy:cy+ch, cx:cx+cw]
            
            if roi.size == 0:
                continue
            
            # Calculate color standard deviation
            std_b = np.std(roi[:, :, 0])
            std_g = np.std(roi[:, :, 1])
            std_r = np.std(roi[:, :, 2])
            avg_std = (std_b + std_g + std_r) / 3
            
            # Filter by uniformity
            if avg_std <= max_std_threshold:
                candidate['color_std'] = avg_std
                filtered.append(candidate)
        
        return filtered
    
    def _cluster_into_rows(
        self,
        candidates: List[Dict],
        expected_rows: Optional[int] = None
    ) -> List[Dict]:
        """
        Cluster candidates into rows based on Y-coordinate.
        
        Args:
            candidates: List of candidate regions
            expected_rows: Hint for expected number of rows
        
        Returns:
            List of row dicts with swatches
        """
        if not candidates:
            return []
        
        # Sort by Y coordinate
        sorted_candidates = sorted(candidates, key=lambda c: c['y'] + c['height'] / 2)
        
        # Cluster by Y proximity
        rows = []
        current_row = [sorted_candidates[0]]
        current_y = sorted_candidates[0]['y'] + sorted_candidates[0]['height'] / 2
        
        for candidate in sorted_candidates[1:]:
            candidate_y = candidate['y'] + candidate['height'] / 2
            
            # Check if same row (within threshold)
            if abs(candidate_y - current_y) <= self.row_clustering_threshold:
                current_row.append(candidate)
                # Update row Y center
                current_y = sum(c['y'] + c['height'] / 2 for c in current_row) / len(current_row)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [candidate]
                current_y = candidate_y
        
        # Don't forget last row
        if current_row:
            rows.append(current_row)
        
        # Sort each row by X coordinate (left to right)
        for row in rows:
            row.sort(key=lambda c: c['x'])
        
        # Format as row dicts
        formatted_rows = []
        for i, row in enumerate(rows):
            y_center = int(sum(c['y'] + c['height'] / 2 for c in row) / len(row))
            formatted_rows.append({
                'row_index': i,
                'y_center': y_center,
                'swatches': row  # Will be replaced with formatted swatches
            })
        
        return formatted_rows
    
    def _extract_colors(
        self,
        image: np.ndarray,
        lab_image: np.ndarray,
        rows: List[Dict]
    ) -> List[Dict]:
        """
        Extract LAB colors from each swatch region.
        
        Args:
            image: Original BGR image
            lab_image: LAB-converted image
            rows: Row dicts with swatch candidates
        
        Returns:
            Rows with color information added
        """
        result_rows = []
        
        for row in rows:
            formatted_swatches = []
            
            for candidate in row['swatches']:
                x, y, w, h = candidate['x'], candidate['y'], candidate['width'], candidate['height']
                
                # Extract LAB color from center region
                margin = max(2, min(w, h) // 4)
                cx = x + margin
                cy = y + margin
                cw = max(1, w - 2 * margin)
                ch = max(1, h - 2 * margin)
                
                try:
                    lab_values = extract_lab_values(lab_image, (cx, cy, cw, ch))
                    
                    # Get RGB for hex conversion
                    roi = image[cy:cy+ch, cx:cx+cw]
                    mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
                    r, g, b = int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0])
                    hex_color = rgb_to_hex(r, g, b)
                    
                    formatted_swatches.append({
                        'region': {
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h
                        },
                        'color': {
                            'lab': {
                                'L': round(lab_values['L'], 2),
                                'a': round(lab_values['a'], 2),
                                'b': round(lab_values['b'], 2)
                            },
                            'hex': hex_color
                        }
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to extract color at ({x}, {y}): {e}")
                    continue
            
            result_rows.append({
                'row_index': row['row_index'],
                'y_center': row['y_center'],
                'swatches': formatted_swatches
            })
        
        return result_rows
    
    def _success_result(
        self,
        rows: List[Dict],
        width: int,
        height: int,
        start_time: float
    ) -> Dict:
        """Create successful result with empty rows."""
        import time
        return {
            'success': True,
            'data': {
                'rows': rows,
                'total_swatches': 0,
                'image_dimensions': {'width': width, 'height': height},
                'processing_time_ms': int((time.time() - start_time) * 1000)
            }
        }


# Legacy alias for backwards compatibility
ReferenceSquareDetectionService = SwatchDetectionService
