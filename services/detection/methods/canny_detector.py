"""
Canny edge detection method for strip detection.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .base_detector import BaseStripDetector

logger = logging.getLogger(__name__)


class CannyDetector(BaseStripDetector):
    """Detect strip using Canny edge detection method."""
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[Dict], str, float]:
        """
        Detect strip using Canny edge detection method.
        
        Step-by-step pipeline:
        1. Convert to grayscale
        2. Apply Gaussian blur
        3. Run Canny edge detection
        4. Find contours
        5. Filter contours by geometric rules (area, aspect ratio)
        6. Select largest qualifying rectangle
        7. Extract strip using minAreaRect → boxPoints
        8. Warp/deskew via perspective transform
        9. Normalize brightness (LAB color space)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (strip_region, orientation, angle) or (None, 'vertical', 0.0) if not found
        """
        h, w = image.shape[:2]
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.visual_logger:
            self.visual_logger.add_step('canny_01_gray', 'Grayscale conversion', 
                                       cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        
        # Step 2: Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.visual_logger:
            self.visual_logger.add_step('canny_02_blur', 'Gaussian blur (5x5)', 
                                       cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR))
        
        # Step 3: Run Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        if self.visual_logger:
            self.visual_logger.add_step('canny_03_edges', 'Canny edge detection (50, 150)', 
                                       cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.visual_logger:
            vis_contours = image.copy()
            cv2.drawContours(vis_contours, contours, -1, (0, 255, 0), 2)
            cv2.putText(vis_contours, f'Found {len(contours)} contours', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.visual_logger.add_step('canny_04_contours', f'All contours ({len(contours)} found)', 
                                       vis_contours)
        
        # Step 5: Filter contours using geometric rules
        min_area = max(5000, int(w * h * 0.01))  # At least 5000px or 1% of image
        min_aspect_ratio = 6.0
        max_aspect_ratio = 20.0
        
        candidate_rectangles = []
        vis_filtered = image.copy()
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(cnt)
            (rect_w, rect_h) = rect[1]
            rect_angle = rect[2]
            
            if rect_w == 0 or rect_h == 0:
                continue
            
            # Calculate aspect ratio (longer / shorter)
            aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
            
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # Valid candidate
            candidate_rectangles.append((rect, area, aspect_ratio))
            
            # Visualize candidate
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(vis_filtered, [box], 0, (0, 255, 0), 2)
            cv2.putText(vis_filtered, f'C{i}: AR={aspect_ratio:.1f}, A={area:.0f}', 
                       (int(box[0][0]), int(box[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.visual_logger:
            cv2.putText(vis_filtered, f'Filtered: {len(candidate_rectangles)} candidates', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.visual_logger.add_step('canny_05_filtered', 
                                       f'Filtered candidates ({len(candidate_rectangles)} valid)', 
                                       vis_filtered)
        
        # Step 6: Select the largest qualifying rectangle
        if not candidate_rectangles:
            self.logger.debug("No qualifying rectangles found in Canny method")
            return (None, 'vertical', 0.0)
        
        # Sort by area (largest first)
        candidate_rectangles.sort(key=lambda x: x[1], reverse=True)
        strip_rect, area, aspect_ratio = candidate_rectangles[0]
        
        if self.visual_logger:
            vis_selected = image.copy()
            box = cv2.boxPoints(strip_rect)
            box = np.int0(box)
            cv2.drawContours(vis_selected, [box], 0, (255, 0, 0), 3)
            cv2.putText(vis_selected, f'Selected: AR={aspect_ratio:.1f}, A={area:.0f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            self.visual_logger.add_step('canny_06_selected', 'Selected strip rectangle', vis_selected)
        
        # Step 7: Extract strip using minAreaRect → boxPoints
        box = cv2.boxPoints(strip_rect)
        box = np.float32(box)
        
        # Determine orientation and angle
        (rect_w, rect_h) = strip_rect[1]
        angle = strip_rect[2]
        
        # Normalize angle to -90 to 90 range
        if angle < -45:
            angle += 90
            rect_w, rect_h = rect_h, rect_w
        
        orientation = 'vertical' if rect_h > rect_w else 'horizontal'
        
        # Step 8: Warp/deskew via perspective transform
        # Calculate destination points for upright rectangle
        width = int(max(rect_w, rect_h))
        height = int(min(rect_w, rect_h))
        
        # Ensure minimum dimensions
        width = max(width, 100)
        height = max(height, 20)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        # Sort by sum and difference to get corners
        sum_pts = box.sum(axis=1)
        diff_pts = np.diff(box, axis=1)
        
        pts_src = np.float32([
            box[np.argmin(sum_pts)],      # top-left
            box[np.argmin(diff_pts)],     # top-right
            box[np.argmax(sum_pts)],      # bottom-right
            box[np.argmax(diff_pts)]      # bottom-left
        ])
        
        pts_dst = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ])
        
        try:
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(image, M, (width, height))
            
            if self.visual_logger:
                self.visual_logger.add_step('canny_07_warped', 
                                           f'Warped strip ({width}x{height}, angle={angle:.1f}°)', 
                                           warped)
        except Exception as e:
            self.logger.error(f"Perspective transform failed: {e}")
            # Fallback: use bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(np.int0(box))
            warped = image[y:y+h_rect, x:x+w_rect]
            if warped.size == 0:
                return (None, 'vertical', 0.0)
        
        # Step 9: Normalize brightness (LAB color space)
        normalized = self._normalize_brightness_lab(warped)
        if self.visual_logger:
            self.visual_logger.add_step('canny_08_normalized', 'Brightness normalized (LAB)', 
                                      normalized)
        
        # Convert warped strip region back to original image coordinates
        # For now, use the bounding box of the original rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(np.int0(box))
        
        strip_region = {
            'top': max(0, y),
            'bottom': min(h, y + h_rect),
            'left': max(0, x),
            'right': min(w, x + w_rect)
        }
        
        return (strip_region, orientation, angle)
    
    def _normalize_brightness_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize brightness using LAB color space.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Brightness-normalized image (BGR format)
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0].astype(np.float32)
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Find white reference (95th percentile of L channel)
        white_reference = np.percentile(L_channel, 95)
        target_l = 100.0  # Target L value for white
        
        if white_reference > 0:
            # Scale L channel to normalize brightness
            scale_factor = target_l / white_reference
            L_normalized = np.clip(L_channel * scale_factor, 0, 100)
        else:
            L_normalized = L_channel
        
        # Convert back to uint8
        L_normalized = L_normalized.astype(np.uint8)
        
        # Reconstruct LAB image
        lab_normalized = cv2.merge([L_normalized, a_channel, b_channel])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        
        return normalized

