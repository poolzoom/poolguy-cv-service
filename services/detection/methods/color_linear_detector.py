"""
Color-based linear structure detector using LAB L-channel continuity.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List

from .base_detector import BaseStripDetector

logger = logging.getLogger(__name__)


class ColorLinearDetector(BaseStripDetector):
    """
    Detect strip using LAB L-channel (lightness) continuity.
    
    The strip has 3 properties:
    1. It is the brightest continuous vertical region in the image
    2. It is a long, vertical, continuous column of high-L pixels
    3. The pads create periodic color blocks along that column
    """
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[Dict], str, float]:
        """
        Detect strip using LAB L-channel continuity method.
        
        Algorithm:
        1. Convert image to LAB color space
        2. Extract L (lightness) channel
        3. Find brightest continuous vertical region (project horizontally)
        4. Expand to find continuous high-L region
        5. Validate with periodic color blocks (pads)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            (strip_region, orientation, angle) or (None, 'vertical', 0.0) if not found
        """
        h, w = image.shape[:2]
        
        # Step 1: Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0]
        
        if self.visual_logger:
            self.visual_logger.add_step('color_01_lab', 'LAB color space conversion', 
                                       cv2.cvtColor(L_channel, cv2.COLOR_GRAY2BGR))
            self.visual_logger.add_step('color_02_L_channel', 'L (lightness) channel', 
                                       cv2.cvtColor(L_channel, cv2.COLOR_GRAY2BGR))
        
        # Step 2: Project L values horizontally (sum columns)
        # This finds the brightest vertical column
        horizontal_projection = np.sum(L_channel, axis=0)  # Sum along rows (vertical)
        
        if self.visual_logger:
            # Visualize projection
            vis_proj = image.copy()
            max_proj = np.max(horizontal_projection)
            if max_proj > 0:
                proj_normalized = (horizontal_projection / max_proj * h).astype(np.int32)
                for x in range(w):
                    cv2.line(vis_proj, (x, h), (x, h - proj_normalized[x]), (0, 255, 0), 1)
            cv2.putText(vis_proj, f'Horizontal projection (max={max_proj:.0f})', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.visual_logger.add_step('color_03_projection', 'Horizontal projection of L values', vis_proj)
        
        # Step 3: Find peak in projection (brightest column)
        peak_x = np.argmax(horizontal_projection)
        peak_value = horizontal_projection[peak_x]
        
        # Step 4: Expand to find continuous high-L region
        # Threshold: at least 80% of peak value
        threshold = peak_value * 0.80
        
        # Find all columns above threshold
        above_threshold = horizontal_projection >= threshold
        
        # Find continuous region around peak
        # Start from peak and expand left and right
        left_bound = peak_x
        right_bound = peak_x
        
        # Expand left
        while left_bound > 0 and above_threshold[left_bound - 1]:
            left_bound -= 1
        
        # Expand right
        while right_bound < w - 1 and above_threshold[right_bound + 1]:
            right_bound += 1
        
        # Step 5: Find vertical extent (top and bottom)
        # Extract the strip region and find where high-L pixels are
        strip_roi_L = L_channel[:, left_bound:right_bound + 1]
        
        if strip_roi_L.size == 0:
            self.logger.debug("No strip region found in color linear method")
            return (None, 'vertical', 0.0)
        
        # Project vertically to find top and bottom
        vertical_projection = np.sum(strip_roi_L, axis=1)  # Sum along columns (horizontal)
        
        # Find peak and mean to determine strip region
        max_vert_proj = np.max(vertical_projection)
        mean_vert_proj = np.mean(vertical_projection)
        if max_vert_proj == 0:
            return (None, 'vertical', 0.0)
        
        # Smooth the projection to reduce noise
        kernel_size = max(5, len(vertical_projection) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_proj = cv2.GaussianBlur(vertical_projection.reshape(-1, 1), (kernel_size, 1), 0).flatten()
        
        # Find where projection transitions from low to high (top edge) and high to low (bottom edge)
        # Use percentile-based approach to find consistently high region
        proj_sorted = np.sort(smoothed_proj)
        percentile_50 = proj_sorted[int(len(proj_sorted) * 0.50)]  # Median
        percentile_75 = proj_sorted[int(len(proj_sorted) * 0.75)]
        
        # Use 50th percentile as base threshold (median value)
        # This should capture the strip region where values are consistently above median
        strip_threshold = percentile_50
        
        # Find top edge: scan from top down to find where projection rises above threshold
        top = 0
        for y in range(len(smoothed_proj)):
            if smoothed_proj[y] >= strip_threshold:
                top = y
                break
        
        # Find bottom edge: scan from bottom up to find where projection drops below threshold
        bottom = len(smoothed_proj) - 1
        for y in range(len(smoothed_proj) - 1, -1, -1):
            if smoothed_proj[y] >= strip_threshold:
                bottom = y + 1  # Include this row
                break
        
        # Validate we found a reasonable strip
        strip_height = bottom - top
        min_height = int(h * 0.20)
        max_height = int(h * 0.85)  # Reject if it's almost the full height
        
        if strip_height < min_height:
            # Too small - expand around peak
            center_y = (top + bottom) // 2 if (bottom - top) > 0 else peak_y
            top = max(0, center_y - min_height // 2)
            bottom = min(h, center_y + min_height // 2)
        elif strip_height > max_height:
            # Too large - use tighter threshold (60% above mean)
            tight_threshold = mean_vert_proj + (max_vert_proj - mean_vert_proj) * 0.6
            peak_y = np.argmax(smoothed_proj)
            top_tight = top
            for y in range(peak_y, -1, -1):
                if smoothed_proj[y] < tight_threshold:
                    top_tight = y + 1
                    break
            bottom_tight = bottom
            for y in range(peak_y, len(smoothed_proj)):
                if smoothed_proj[y] < tight_threshold:
                    bottom_tight = y
                    break
            if (bottom_tight - top_tight) >= min_height:
                top = max(0, top_tight)
                bottom = min(h, bottom_tight)
        
        top = max(0, top)
        bottom = min(h, bottom)
        
        # Visualize vertical projection for debugging
        if self.visual_logger:
            vis_proj_vert = image.copy()
            # Draw projection as horizontal lines
            max_proj_val = np.max(smoothed_proj)
            if max_proj_val > 0:
                for y in range(len(smoothed_proj)):
                    x_val = int((smoothed_proj[y] / max_proj_val) * w)
                    cv2.line(vis_proj_vert, (0, y), (x_val, y), (0, 255, 0), 1)
            # Mark detected edges
            cv2.line(vis_proj_vert, (0, top), (w, top), (255, 0, 0), 3)
            cv2.line(vis_proj_vert, (0, bottom), (w, bottom), (255, 0, 0), 3)
            cv2.putText(vis_proj_vert, f'Top: {top}, Bottom: {bottom}, Height: {bottom-top}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            self.visual_logger.add_step('color_03b_vertical_projection', 
                                       f'Vertical projection (top={top}, bottom={bottom})', 
                                       vis_proj_vert)
        
        strip_region = {
            'top': top,
            'bottom': bottom,
            'left': left_bound,
            'right': right_bound + 1
        }
        
        if self.visual_logger:
            vis_candidate = image.copy()
            cv2.rectangle(vis_candidate, 
                         (strip_region['left'], strip_region['top']),
                         (strip_region['right'], strip_region['bottom']),
                         (0, 255, 0), 3)
            cv2.putText(vis_candidate, f'Candidate: {strip_region["right"]-strip_region["left"]}x{strip_region["bottom"]-strip_region["top"]}',
                       (strip_region['left'], strip_region['top'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.visual_logger.add_step('color_04_candidate_region', 'Detected bright region', vis_candidate)
        
        # Step 6: Validate with periodic color blocks (pads)
        # Extract strip region and look for periodic variations
        strip_roi = image[strip_region['top']:strip_region['bottom'],
                         strip_region['left']:strip_region['right']]
        
        if strip_roi.size == 0:
            return (None, 'vertical', 0.0)
        
        # Convert to LAB for pad detection
        strip_lab = cv2.cvtColor(strip_roi, cv2.COLOR_BGR2LAB)
        strip_L = strip_lab[:, :, 0]
        
        # Project horizontally to find pad boundaries (vertical divisions)
        # Look for dips in the projection (gaps between pads)
        roi_h, roi_w = strip_L.shape[:2]
        pad_projection = np.sum(strip_L, axis=1)  # Sum along width
        
        # Smooth the projection
        kernel_size = max(3, roi_h // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed_proj = cv2.GaussianBlur(pad_projection.reshape(-1, 1), (kernel_size, 1), 0).flatten()
        
        # Find local minima (potential pad boundaries)
        mean_proj = np.mean(smoothed_proj)
        std_proj = np.std(smoothed_proj)
        threshold_dip = mean_proj - std_proj * 0.5
        
        # Count significant dips
        dips = np.sum(smoothed_proj < threshold_dip)
        
        if self.visual_logger:
            vis_pads = strip_roi.copy()
            # Draw projection
            max_proj_val = np.max(smoothed_proj)
            if max_proj_val > 0:
                for y in range(roi_h):
                    x_val = int((smoothed_proj[y] / max_proj_val) * roi_w)
                    cv2.line(vis_pads, (0, y), (x_val, y), (255, 0, 0), 1)
                    if smoothed_proj[y] < threshold_dip:
                        cv2.line(vis_pads, (0, y), (roi_w, y), (0, 0, 255), 2)
            
            cv2.putText(vis_pads, f'Pad validation: {dips} dips found', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.visual_logger.add_step('color_05_pad_validation', f'Periodic pad detection ({dips} dips)', vis_pads)
        
        # Validate: should have at least 2 dips for 4+ pads
        if dips < 2:
            self.logger.debug(f"Insufficient pad structure detected: {dips} dips")
            # Don't fail completely, but log debug
        
        # Calculate aspect ratio to determine orientation
        strip_width = strip_region['right'] - strip_region['left']
        strip_height = strip_region['bottom'] - strip_region['top']
        
        orientation = 'vertical' if strip_height > strip_width else 'horizontal'
        angle = 0.0  # Assume no rotation for now (can be enhanced later)
        
        if self.visual_logger:
            vis_final = image.copy()
            cv2.rectangle(vis_final,
                         (strip_region['left'], strip_region['top']),
                         (strip_region['right'], strip_region['bottom']),
                         (0, 255, 0), 3)
            cv2.putText(vis_final, f'Final: {orientation}, {strip_width}x{strip_height}',
                       (strip_region['left'], strip_region['top'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.visual_logger.add_step('color_06_final', 'Final strip region', vis_final)
        
        return (strip_region, orientation, angle)

