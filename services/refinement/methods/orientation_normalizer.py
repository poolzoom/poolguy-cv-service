"""
Orientation normalization for strip refinement.

Rotates strip to vertical orientation using edge detection.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from services.refinement.base_refiner import BaseRefinementMethod
from services.utils.debug import DebugContext
from services.refinement.methods.adaptive_params import AdaptiveParameterCalculator

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class OrientationNormalizer(BaseRefinementMethod):
    """Normalize strip orientation to vertical."""
    
    def refine(
        self,
        image: np.ndarray,
        input_region: Optional[Dict] = None,
        debug: Optional[DebugContext] = None
    ) -> Tuple[Dict, Dict]:
        """
        Normalize orientation by rotating strip to vertical.
        
        Only rotates the cropped region, not the whole image.
        Constrains rotation to small angles (max configurable, default 10 degrees).
        
        Args:
            image: Input image (BGR format)
            input_region: Region dict to rotate (required)
            debug: Optional debug context
            
        Returns:
            Tuple of (region_dict, metadata)
            - region_dict: Updated region after rotation (if rotation occurred)
            - metadata: {'angle': float, 'method': str, 'rotated_crop': np.ndarray}
        """
        if not input_region:
            # No region provided, return original
            return {
                'left': 0,
                'top': 0,
                'right': image.shape[1],
                'bottom': image.shape[0]
            }, {'angle': 0.0, 'method': 'none'}
        
        method = self.config.get('method', 'minarearect')  # Default to minAreaRect
        
        # Both methods now use minAreaRect (more reliable)
        if method in ['pca', 'hough', 'minarearect']:
            angle, rotated_crop, expanded_region = self._normalize_orientation(image, input_region, debug)
        else:
            raise ValueError(f"Unknown orientation method: {method}. Use 'pca', 'hough', or 'minarearect'")
        
        # Constrain angle to reasonable range (configurable max)
        max_angle = self.config.get('max_rotation_angle', 30.0)
        min_rotation = self.config.get('min_rotation_threshold', 0.1)
        
        if abs(angle) > max_angle:
            angle = max_angle if angle > 0 else -max_angle
        
        # Apply minimum threshold - don't rotate if angle is too small
        if abs(angle) < min_rotation:
            angle = 0.0
        
        metadata = {
            'angle': angle,
            'method': method,
            'rotated_crop': rotated_crop,  # Only the cropped region, not full image
            'expanded_region': expanded_region  # Return the expanded region coordinates for proper offset calculation
        }
        
        # Return expanded region (not original) since that's what we rotated
        # This ensures subsequent steps use the correct offset
        region = expanded_region.copy()
        
        return region, metadata
    
    def _detect_edges_single_attempt(
        self,
        gray: np.ndarray,
        h_crop: int,
        w_crop: int,
        debug: Optional[DebugContext]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Single attempt at edge detection with adaptive or fixed parameters.
        
        Returns:
            Tuple of (edges, lines, params_used)
        """
        param_calc = AdaptiveParameterCalculator()
        
        # Calculate Canny parameters
        use_adaptive_canny = self.config.get('use_adaptive_canny', True)
        if use_adaptive_canny:
            canny_low, canny_high = param_calc.calculate_canny_params(gray)
            # Allow config override for fine-tuning
            canny_low = self.config.get('canny_low', canny_low)
            canny_high = self.config.get('canny_high', canny_high)
        else:
            canny_low = self.config.get('canny_low', 30)
            canny_high = self.config.get('canny_high', 100)
        
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # Calculate edge density for Hough parameter adjustment
        edge_density = np.sum(edges > 0) / (h_crop * w_crop)
        
        # Calculate Hough parameters
        use_adaptive_hough = self.config.get('use_adaptive_hough', True)
        if use_adaptive_hough:
            hough_params = param_calc.calculate_hough_params(h_crop, w_crop, edge_density)
            min_line_length = hough_params['min_line_length']
            hough_threshold = hough_params['threshold']
            max_line_gap = hough_params['max_line_gap']
            
            # Allow config override for fine-tuning
            if 'hough_min_line_length_ratio' in self.config:
                min_line_length_abs = self.config.get('hough_min_line_length_absolute', 60)
                min_line_length_ratio = self.config.get('hough_min_line_length_ratio', 0.6)
                min_line_length = max(min_line_length_abs, int(h_crop * min_line_length_ratio))
            if 'hough_threshold_ratio' in self.config:
                hough_threshold_abs = self.config.get('hough_threshold_absolute', 20)
                hough_threshold_ratio = self.config.get('hough_threshold_ratio', 0.05)
                hough_threshold = max(hough_threshold_abs, int(h_crop * hough_threshold_ratio))
            if 'hough_max_line_gap' in self.config:
                max_line_gap = self.config['hough_max_line_gap']
        else:
            # Use config-based parameters (current behavior)
            min_line_length_ratio = self.config.get('hough_min_line_length_ratio', 0.5)
            min_line_length_abs = self.config.get('hough_min_line_length_absolute', 100)
            min_line_length = max(min_line_length_abs, int(h_crop * min_line_length_ratio))
            
            hough_threshold_ratio = self.config.get('hough_threshold_ratio', 0.15)
            hough_threshold_abs = self.config.get('hough_threshold_absolute', 50)
            hough_threshold = max(hough_threshold_abs, int(h_crop * hough_threshold_ratio))
            
            max_line_gap = self.config.get('hough_max_line_gap', 5)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        params_used = {
            'canny_low': canny_low,
            'canny_high': canny_high,
            'min_line_length': min_line_length,
            'hough_threshold': hough_threshold,
            'max_line_gap': max_line_gap,
            'edge_density': float(edge_density),
            'lines_detected': len(lines) if lines is not None else 0,
            'adaptive_canny': use_adaptive_canny,
            'adaptive_hough': use_adaptive_hough
        }
        
        if debug:
            debug.add_step(
                'edge_detection',
                'Edge Detection (single attempt)',
                edges,
                params_used
            )
        
        return edges, lines, params_used
    
    def _detect_with_adaptive_retry(
        self,
        gray: np.ndarray,
        h_crop: int,
        w_crop: int,
        debug: Optional[DebugContext]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Try edge detection with adaptive parameters, retry with relaxed params if needed.
        
        Returns:
            Tuple of (edges, lines, params_used)
        """
        param_calc = AdaptiveParameterCalculator()
        
        # Try up to 3 parameter sets (strict -> moderate -> relaxed)
        param_sets = [
            {'name': 'strict', 'canny_factor': 1.0, 'hough_factor': 1.0},
            {'name': 'moderate', 'canny_factor': 0.8, 'hough_factor': 0.7},
            {'name': 'relaxed', 'canny_factor': 0.6, 'hough_factor': 0.5}
        ]
        
        for param_set in param_sets:
            # Calculate base parameters
            canny_low, canny_high = param_calc.calculate_canny_params(gray)
            canny_low = int(canny_low * param_set['canny_factor'])
            canny_high = int(canny_high * param_set['canny_factor'])
            
            edges = cv2.Canny(gray, canny_low, canny_high)
            edge_density = np.sum(edges > 0) / (h_crop * w_crop)
            
            hough_params = param_calc.calculate_hough_params(h_crop, w_crop, edge_density)
            min_line_length = int(hough_params['min_line_length'] * param_set['hough_factor'])
            hough_threshold = int(hough_params['threshold'] * param_set['hough_factor'])
            max_line_gap = int(hough_params['max_line_gap'] * 1.5)  # More gap tolerance
            
            # Apply config overrides if present
            if 'hough_min_line_length_ratio' in self.config:
                min_line_length_abs = self.config.get('hough_min_line_length_absolute', 60)
                min_line_length_ratio = self.config.get('hough_min_line_length_ratio', 0.6)
                base_min_length = max(min_line_length_abs, int(h_crop * min_line_length_ratio))
                min_line_length = int(base_min_length * param_set['hough_factor'])
            
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=hough_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
            
            lines_count = len(lines) if lines is not None else 0
            
            params_used = {
                'set': param_set['name'],
                'canny_low': canny_low,
                'canny_high': canny_high,
                'min_line_length': min_line_length,
                'hough_threshold': hough_threshold,
                'max_line_gap': max_line_gap,
                'edge_density': float(edge_density),
                'lines_detected': lines_count
            }
            
            if debug:
                debug.add_step(
                    f'edge_detection_{param_set["name"]}',
                    f'Edge Detection ({param_set["name"]} params)',
                    edges,
                    params_used
                )
            
            # If we found lines, use this parameter set
            if lines_count > 0:
                return edges, lines, params_used
        
        # All parameter sets failed - return last attempt
        return edges, None, params_used
    
    def _normalize_orientation(
        self,
        image: np.ndarray,
        region: Dict,
        debug: Optional[DebugContext]
    ) -> Tuple[float, np.ndarray, Dict]:
        """
        Detect and correct strip rotation using bounding box aspect ratio as primary method,
        with Hough lines as refinement.
        
        This method:
        1. Estimates rotation from bounding box aspect ratio (primary)
        2. Optionally refines using Hough line detection (secondary)
        3. Rotates the cropped region to make strip vertical
        """
        # Crop to region, but expand by configurable ratio to catch edges slightly outside YOLO bbox
        x1, y1 = region['left'], region['top']
        x2, y2 = region['right'], region['bottom']
        
        # Calculate expansion (adaptive based on aspect ratio - more rotation needs more expansion)
        width = x2 - x1
        height = y2 - y1
        base_expand_ratio = self.config.get('expand_ratio', 0.05)
        
        # Calculate aspect ratio to estimate rotation
        # Lower aspect ratio (wider bbox) suggests more rotation, needs more expansion
        aspect_ratio = height / width if width > 0 else 0
        expected_aspect_ratio = 7.0  # Expected for vertical strip
        
        # If aspect ratio is much lower than expected, strip is likely rotated significantly
        # Expand more to catch edges that extend outside the bbox
        if aspect_ratio < expected_aspect_ratio * 0.7:
            # Significant rotation detected - expand more
            rotation_factor = (expected_aspect_ratio * 0.7) / max(aspect_ratio, 0.1)
            expand_ratio = base_expand_ratio * min(rotation_factor, 3.0)  # Cap at 3x base ratio
        else:
            expand_ratio = base_expand_ratio
        
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        
        # Expand region with bounds checking
        h_img, w_img = image.shape[:2]
        x1_expanded = max(0, x1 - expand_x)
        y1_expanded = max(0, y1 - expand_y)
        x2_expanded = min(w_img, x2 + expand_x)
        y2_expanded = min(h_img, y2 + expand_y)
        
        cropped = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded].copy()
        
        if cropped.size == 0:
            expanded_region = {
                'left': x1_expanded,
                'top': y1_expanded,
                'right': x2_expanded,
                'bottom': y2_expanded
            }
            return 0.0, cropped, expanded_region
        
        h_crop, w_crop = cropped.shape[:2]
        
        # Adjust expected edge positions for expanded region
        # Original region offset within expanded crop
        offset_x = x1 - x1_expanded
        offset_y = y1 - y1_expanded
        
        # VALIDATION METHOD: Use bbox aspect ratio as a guide/validation
        # We don't know the actual strip dimensions, so we can't directly calculate rotation
        # But we can use the aspect ratio to validate that detected angles are reasonable
        original_width = width
        original_height = height
        current_aspect_ratio = original_height / original_width if original_width > 0 else 0
        
        # Expected aspect ratio for a vertical strip (height/width)
        # Typical strips are 5-10x taller than wide when vertical
        expected_aspect_ratio_vertical = 7.0  # Middle of typical range
        
        # If current aspect ratio is much lower than expected, strip is likely rotated
        # This gives us a rough guide for what rotation range to expect
        aspect_ratio_suggests_rotation = current_aspect_ratio < expected_aspect_ratio_vertical * 0.7
        
        # Estimate rough rotation range from aspect ratio (for validation only)
        # If aspect ratio is low, rotation is likely significant (>10°)
        # If aspect ratio is close to expected, rotation is likely small (<5°)
        if aspect_ratio_suggests_rotation:
            estimated_rotation_range = (10.0, 25.0)  # Likely 10-25 degrees
        else:
            estimated_rotation_range = (0.0, 10.0)  # Likely 0-10 degrees
        
        # PRIMARY METHOD: Detect rotation using Hough lines
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive parameters with iterative retry if enabled
        use_iterative_retry = self.config.get('use_iterative_retry', True)
        if use_iterative_retry:
            edges, lines, params_used = self._detect_with_adaptive_retry(
                gray, h_crop, w_crop, debug
            )
        else:
            # Single attempt with adaptive or fixed parameters
            edges, lines, params_used = self._detect_edges_single_attempt(
                gray, h_crop, w_crop, debug
            )
        
        # Extract min_line_length from params_used for line filtering
        min_line_length = params_used.get('min_line_length', 100)
        
        hough_rotation_angle = 0.0
        detected_lines = []
        all_detected_lines = []  # All filtered lines for visualization
        best_angle = None
        
        if lines is not None and len(lines) > 0:
            # Estimate expected edge positions from original YOLO bbox within expanded crop
            # Account for the expansion offset
            original_left_in_crop = offset_x
            original_right_in_crop = offset_x + width
            expected_left_x = original_left_in_crop
            expected_right_x = original_right_in_crop
            edge_tolerance_x = width * 0.3  # 30% of original width tolerance
            
            # Filter for near-vertical lines (strip edges) with SPATIAL CONSTRAINTS
            vertical_lines = []
            
            for line in lines:
                x1_line, y1_line, x2_line, y2_line = line[0]
                dx = x2_line - x1_line
                dy = y2_line - y1_line
                
                if abs(dx) < 2:  # Skip perfectly vertical lines (no rotation info)
                    continue
                
                # Calculate angle from horizontal
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                # Normalize to -90 to 90 range
                if angle_deg > 90:
                    angle_deg -= 180
                elif angle_deg < -90:
                    angle_deg += 180
                
                # Configurable angle tolerance
                angle_tolerance = self.config.get('angle_tolerance_degrees', 15.0)
                angle_from_vertical = abs(abs(angle_deg) - 90)
                if angle_from_vertical > angle_tolerance:
                    continue
                
                # Check line length - prefer longer lines
                line_length = np.sqrt(dx**2 + dy**2)
                if line_length < min_line_length:
                    continue
                
                # Configurable spatial filter
                use_spatial_filter = self.config.get('use_spatial_filter', True)
                spatial_tolerance_ratio = self.config.get('spatial_tolerance_ratio', 0.4)
                edge_tolerance_x = width * spatial_tolerance_ratio
                
                if use_spatial_filter:
                    x_center = (x1_line + x2_line) / 2
                    is_near_left = abs(x_center - expected_left_x) < edge_tolerance_x
                    is_near_right = abs(x_center - expected_right_x) < edge_tolerance_x
                    
                    if not (is_near_left or is_near_right):
                        continue  # Skip lines that aren't near expected edges
                
                vertical_lines.append({
                    'line': line[0],
                    'angle': angle_deg,
                    'x_center': x_center,
                    'length': line_length,
                    'distance_from_expected': min(
                        abs(x_center - expected_left_x),
                        abs(x_center - expected_right_x)
                    )
                })
            
            if len(vertical_lines) > 0:
                # Use single best line - prioritize longest line (most reliable edge)
                # Sort by length (longest first)
                vertical_lines.sort(key=lambda v: v['length'], reverse=True)
                best_line = vertical_lines[0]
                
                def edge_angle_to_rotation(edge_angle):
                    """Convert edge angle to rotation angle needed."""
                    if abs(edge_angle) > 45:
                        if edge_angle < 0:
                            return 90 + edge_angle
                        else:
                            return 90 - edge_angle
                    else:
                        return -edge_angle
                
                # Use the longest line's angle directly
                best_angle = best_line['angle']
                hough_rotation_angle = edge_angle_to_rotation(best_angle)
                
                # Store for visualization
                detected_lines = [best_line['line']]
                all_detected_lines = [v['line'] for v in vertical_lines]  # All filtered lines
                
                # For debug output (backward compatibility)
                left_lines = []
                right_lines = []
                left_avg_angle = None
                right_avg_angle = None
                left_rotation = 0.0
                right_rotation = 0.0
        
        # Use Hough as primary method, validate with bbox aspect ratio
        rotation_angle = hough_rotation_angle
        
        # VALIDATION: Check if detected angle is reasonable given bbox aspect ratio
        if abs(hough_rotation_angle) > 1.0:
            # Hough detected rotation - validate it makes sense
            detected_in_range = (estimated_rotation_range[0] <= abs(hough_rotation_angle) <= estimated_rotation_range[1])
            
            if not detected_in_range and aspect_ratio_suggests_rotation:
                # Detected angle seems too small given the low aspect ratio
                # This might indicate under-rotation - trust Hough but note the discrepancy
                # (We'll still use Hough, but this helps with debugging)
                pass
        else:
            # Hough didn't detect rotation - check if bbox suggests we should have
            if aspect_ratio_suggests_rotation:
                # Bbox suggests rotation but Hough didn't find it
                # This is a problem - Hough should have found edges
                # Fall through to minAreaRect fallback
                pass
        
        # Fallback to minAreaRect if both methods failed
        if abs(rotation_angle) < 0.5:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                min_contour_area = self.config.get('min_contour_area', 100)
                if cv2.contourArea(largest_contour) >= min_contour_area:
                    rect = cv2.minAreaRect(largest_contour)
                    (rect_w, rect_h), angle_rect = rect[1], rect[2]
                    
                    # Normalize minAreaRect angle
                    if angle_rect < -45:
                        angle_rect += 90
                    rotation_angle = -angle_rect
        
        # Apply rotation
        if abs(rotation_angle) > 0.0:
            center = (w_crop // 2, h_crop // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_crop = cv2.warpAffine(cropped, rotation_matrix, (w_crop, h_crop), 
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated_crop = cropped
        
        if debug:
            vis = image.copy()
            # Draw original YOLO bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw expanded region
            cv2.rectangle(vis, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (255, 0, 255), 1)  # Magenta for expanded
            
            # Draw all detected lines (for debugging) - adjust coordinates for expanded region
            for line in all_detected_lines:
                x1_line, y1_line, x2_line, y2_line = line
                cv2.line(vis, (x1_expanded + x1_line, y1_expanded + y1_line), 
                        (x1_expanded + x2_line, y1_expanded + y2_line), (128, 128, 128), 1)  # Gray for all lines
            
            # Draw selected lines (top N from each side) in bright color
            for line in detected_lines:
                x1_line, y1_line, x2_line, y2_line = line
                cv2.line(vis, (x1_expanded + x1_line, y1_expanded + y1_line), 
                        (x1_expanded + x2_line, y1_expanded + y2_line), (255, 255, 0), 3)  # Yellow for selected lines
            
            # Draw text with method info
            text_y = max(30, y1_expanded - 30)
            cv2.putText(vis, f'Hough Detection: {hough_rotation_angle:.2f}°',
                       (x1_expanded, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  # Cyan
            text_y += 25
            cv2.putText(vis, f'Final Rotation: {rotation_angle:.2f}°',
                       (x1_expanded, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green
            text_y += 25
            cv2.putText(vis, f'Aspect Ratio: {current_aspect_ratio:.2f} (exp: {expected_aspect_ratio_vertical:.1f})',
                       (x1_expanded, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Yellow
            text_y += 20
            cv2.putText(vis, f'Expanded: +5% ({expand_x}px, {expand_y}px)',
                       (x1_expanded, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)  # Magenta
            
            debug_data = {
                'rotation_angle': float(rotation_angle),
                'hough_rotation_angle': float(hough_rotation_angle),
                'rotation_combination': self.config.get('rotation_combination', 'average'),
                'bbox_aspect_ratio': float(current_aspect_ratio),
                'expected_aspect_ratio': float(expected_aspect_ratio_vertical),
                'aspect_ratio_suggests_rotation': bool(aspect_ratio_suggests_rotation),
                'estimated_rotation_range': [float(estimated_rotation_range[0]), float(estimated_rotation_range[1])],
                'bbox_width': int(original_width),
                'bbox_height': int(original_height),
                'expanded_width': int(w_crop),
                'expanded_height': int(h_crop),
                'selected_lines_count': int(len(detected_lines)),
                'all_filtered_lines_count': int(len(vertical_lines)) if 'vertical_lines' in locals() else 0,
                'total_hough_lines': int(len(lines)) if lines is not None else 0,
                'left_lines_count': 0,
                'right_lines_count': 0,
                'canny_low': params_used.get('canny_low', 30),
                'canny_high': params_used.get('canny_high', 100),
                'hough_threshold': params_used.get('hough_threshold', 50),
                'min_line_length': params_used.get('min_line_length', min_line_length),
                'angle_tolerance': self.config.get('angle_tolerance_degrees', 15.0),
                'edge_density': params_used.get('edge_density', 0.0),
                'param_set': params_used.get('set', 'single')
            }
            if best_angle is not None:
                debug_data['best_angle'] = float(best_angle)
                debug_data['best_line_length'] = float(vertical_lines[0]['length']) if len(vertical_lines) > 0 else 0.0
                debug_data['best_line_distance'] = float(vertical_lines[0]['distance_from_expected']) if len(vertical_lines) > 0 else 0.0
            
            debug.add_step('orientation_pca', 'Orientation Detection (BBox + Hough)', vis, debug_data)
        
        # Return expanded region coordinates for proper offset calculation in subsequent steps
        expanded_region = {
            'left': x1_expanded,
            'top': y1_expanded,
            'right': x2_expanded,
            'bottom': y2_expanded
        }
        return rotation_angle, rotated_crop, expanded_region
    
    def _normalize_hough(
        self,
        image: np.ndarray,
        region: Dict,
        debug: Optional[DebugContext]
    ) -> Tuple[float, np.ndarray]:
        """Normalize using same method (alias for compatibility)."""
        return self._normalize_orientation(image, region, debug)
