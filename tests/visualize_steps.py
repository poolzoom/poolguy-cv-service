"""
Step-by-step visualization of pad detection pipeline.
Shows each step: strip detection, handle detection, square locations, and colors.
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.pipeline.steps.color_extraction import ColorExtractionService
from utils.image_loader import load_image


def visualize_step_by_step(image_path: str, expected_pad_count: int = 6, output_dir: str = 'tests/fixtures/step_visualizations'):
    """Visualize each step of the detection pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = load_image(image_path)
    h, w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    service = ColorExtractionService()
    
    print(f"\n{'='*70}")
    print(f"VISUALIZING: {base_name}")
    print(f"{'='*70}")
    print(f"Image: {w}x{h} pixels\n")
    
    # Create visualization images for each step
    vis_images = {}
    
    # Step 1: General strip area
    print("Step 1: Detecting general strip area...")
    strip_region = service._detect_strip_general_area(image)
    if strip_region:
        x_approx, y_approx, w_approx, h_approx = strip_region
        print(f"  ✓ Detected: {w_approx}x{h_approx} at ({x_approx}, {y_approx})")
        
        vis_step1 = image.copy()
        cv2.rectangle(vis_step1, (x_approx, y_approx), 
                     (x_approx + w_approx, y_approx + h_approx), 
                     (0, 255, 0), 3)
        cv2.putText(vis_step1, "Step 1: General Strip Area", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        vis_images['step1_general_area'] = vis_step1
    else:
        print("  ✗ Failed")
        return
    
    # Step 1.5: Angle and orientation
    print("\nStep 1.5: Detecting angle and orientation...")
    angle, is_vertical = service._detect_strip_angle_and_orientation(image, strip_region)
    if angle is not None:
        print(f"  ✓ Angle: {angle:.2f}°, Orientation: {'VERTICAL' if is_vertical else 'HORIZONTAL'}")
        
        vis_step1_5 = image.copy()
        cv2.rectangle(vis_step1_5, (x_approx, y_approx), 
                     (x_approx + w_approx, y_approx + h_approx), 
                     (0, 255, 0), 2)
        
        # Draw angle indicator
        center_x = x_approx + w_approx // 2
        center_y = y_approx + h_approx // 2
        length = min(w_approx, h_approx) // 2
        end_x = int(center_x + length * np.cos(np.radians(angle)))
        end_y = int(center_y + length * np.sin(np.radians(angle)))
        cv2.arrowedLine(vis_step1_5, (center_x, center_y), (end_x, end_y),
                       (255, 0, 0), 3, tipLength=0.2)
        
        cv2.putText(vis_step1_5, f"Step 1.5: Angle={angle:.1f}° ({'VERTICAL' if is_vertical else 'HORIZONTAL'})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        vis_images['step1_5_angle'] = vis_step1_5
    else:
        print("  ✗ Failed")
        return
    
    # Step 2: Top/bottom edges
    print("\nStep 2: Detecting top/bottom edges...")
    top_edge, bottom_edge = service._detect_strip_vertical_edges_with_angle(
        image, strip_region, angle, is_vertical
    )
    if top_edge is not None and bottom_edge is not None:
        print(f"  ✓ Top: {top_edge}, Bottom: {bottom_edge}")
        
        vis_step2 = image.copy()
        cv2.line(vis_step2, (0, top_edge), (w, top_edge), (255, 0, 0), 3)
        cv2.line(vis_step2, (0, bottom_edge), (w, bottom_edge), (255, 0, 0), 3)
        cv2.putText(vis_step2, f"Step 2: Top={top_edge}, Bottom={bottom_edge}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        vis_images['step2_vertical_edges'] = vis_step2
    else:
        print("  ✗ Failed")
        return
    
    # Step 3: Left/right edges
    print("\nStep 3: Detecting left/right edges...")
    left_edge, right_edge = service._detect_strip_horizontal_edges_with_angle(
        image, x_approx, top_edge, bottom_edge, angle, is_vertical
    )
    if left_edge is not None and right_edge is not None:
        print(f"  ✓ Left: {left_edge}, Right: {right_edge}")
        print(f"  Strip dimensions: {right_edge - left_edge}x{bottom_edge - top_edge}")
        
        vis_step3 = image.copy()
        cv2.rectangle(vis_step3, (left_edge, top_edge), 
                     (right_edge, bottom_edge), 
                     (0, 0, 255), 3)
        cv2.line(vis_step3, (left_edge, 0), (left_edge, h), (0, 0, 255), 2)
        cv2.line(vis_step3, (right_edge, 0), (right_edge, h), (0, 0, 255), 2)
        cv2.putText(vis_step3, f"Step 3: Left={left_edge}, Right={right_edge}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        vis_images['step3_horizontal_edges'] = vis_step3
    else:
        print("  ✗ Failed")
        return
    
    # Step 3.5: Extract and align strip
    print("\nStep 3.5: Extracting and aligning strip...")
    strip_corners = service._get_strip_corners(left_edge, right_edge, top_edge, bottom_edge, angle)
    aligned_strip, transform_matrix = service._extract_and_align_strip(image, strip_corners)
    
    if aligned_strip is not None:
        aligned_h, aligned_w = aligned_strip.shape[:2]
        print(f"  ✓ Aligned strip: {aligned_w}x{aligned_h}")
        
        # Draw corners on original image
        vis_step3_5 = image.copy()
        for i, corner in enumerate(strip_corners):
            cv2.circle(vis_step3_5, tuple(corner), 10, (255, 255, 0), -1)
            cv2.putText(vis_step3_5, str(i), tuple(corner + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(vis_step3_5, "Step 3.5: Strip Corners & Alignment", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        vis_images['step3_5_alignment'] = vis_step3_5
        vis_images['aligned_strip'] = aligned_strip
    else:
        print("  ✗ Failed")
        return
    
    # Step 4: Detect colored squares
    print("\nStep 4: Detecting colored squares...")
    pad_regions = service._detect_colored_squares_in_strip(
        aligned_strip, expected_pad_count, aligned_w, aligned_h
    )
    print(f"  ✓ Found {len(pad_regions)} squares")
    
    # Visualize squares in aligned strip
    vis_step4 = aligned_strip.copy()
    for i, (x, y, rw, rh) in enumerate(pad_regions):
        cv2.rectangle(vis_step4, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
        cv2.putText(vis_step4, f"Square {i}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_step4, f"Step 4: Detected {len(pad_regions)} Squares", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    vis_images['step4_squares'] = vis_step4
    
    # Step 5: Transform back and show final result
    print("\nStep 5: Transforming coordinates back to original image...")
    final_regions = service._transform_pad_coordinates_to_original(
        pad_regions, transform_matrix
    )
    
    vis_final = image.copy()
    for i, (x, y, rw, rh) in enumerate(final_regions):
        cv2.rectangle(vis_final, (x, y), (x + rw, y + rh), (0, 255, 255), 3)
        cv2.putText(vis_final, f"Pad {i}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis_final, f"Final: {len(final_regions)} Pads Detected", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    vis_images['step5_final'] = vis_final
    
    # Detect handle (longest white section at one end)
    print("\nDetecting handle...")
    handle_end = detect_handle(aligned_strip, is_vertical)
    if handle_end:
        print(f"  ✓ Handle detected at: {handle_end}")
        vis_handle = aligned_strip.copy()
        if handle_end == 'top':
            cv2.rectangle(vis_handle, (0, 0), (aligned_w, aligned_h // 4), 
                         (255, 0, 255), 3)
            cv2.putText(vis_handle, "HANDLE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            cv2.rectangle(vis_handle, (0, aligned_h * 3 // 4), 
                         (aligned_w, aligned_h), (255, 0, 255), 3)
            cv2.putText(vis_handle, "HANDLE", (10, aligned_h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        vis_images['handle'] = vis_handle
    
    # Show predicted square locations
    print("\nPredicting square locations based on strip dimensions...")
    predicted_squares = predict_square_locations(
        aligned_strip, expected_pad_count, handle_end, is_vertical
    )
    
    vis_predicted = aligned_strip.copy()
    for i, (x, y, size) in enumerate(predicted_squares):
        cv2.rectangle(vis_predicted, (x, y), (x + size, y + size), 
                     (255, 165, 0), 2)
        cv2.putText(vis_predicted, f"Pred {i}", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    cv2.putText(vis_predicted, f"Predicted {len(predicted_squares)} Square Locations", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    vis_images['predicted_squares'] = vis_predicted
    
    # Extract and show colors
    print("\nExtracting colors from detected squares...")
    vis_colors = aligned_strip.copy()
    for i, (x, y, rw, rh) in enumerate(pad_regions):
        # Extract average color
        roi = aligned_strip[y:y+rh, x:x+rw]
        avg_color = np.mean(roi.reshape(-1, 3), axis=0).astype(int)
        
        # Draw rectangle with average color
        cv2.rectangle(vis_colors, (x, y), (x + rw, y + rh), 
                     tuple(map(int, avg_color)), -1)
        cv2.rectangle(vis_colors, (x, y), (x + rw, y + rh), (255, 255, 255), 2)
        
        # Show RGB values
        color_text = f"R:{avg_color[2]} G:{avg_color[1]} B:{avg_color[0]}"
        cv2.putText(vis_colors, color_text, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(vis_colors, "Extracted Colors", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    vis_images['colors'] = vis_colors
    
    # Save all visualizations
    print(f"\nSaving visualizations to {output_dir}/...")
    for step_name, vis_img in vis_images.items():
        output_path = os.path.join(output_dir, f"{base_name}_{step_name}.jpg")
        
        # Resize if too large
        max_width = 1920
        if vis_img.shape[1] > max_width:
            scale = max_width / vis_img.shape[1]
            new_w = max_width
            new_h = int(vis_img.shape[0] * scale)
            vis_img = cv2.resize(vis_img, (new_w, new_h))
        
        cv2.imwrite(output_path, vis_img)
        print(f"  ✓ Saved: {step_name}")
    
    print(f"\n{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}\n")


def detect_handle(strip_image: np.ndarray, is_vertical: bool) -> Optional[str]:
    """Detect which end has the handle (longest white section)."""
    h, w = strip_image.shape[:2]
    gray = cv2.cvtColor(strip_image, cv2.COLOR_BGR2GRAY)
    
    if is_vertical:
        # Check top vs bottom
        top_quarter = gray[:h//4, :]
        bottom_quarter = gray[3*h//4:, :]
        
        # Calculate average brightness (white = high)
        top_brightness = np.mean(top_quarter)
        bottom_brightness = np.mean(bottom_quarter)
        
        # Handle is whiter (brighter)
        if top_brightness > bottom_brightness * 1.1:
            return 'top'
        elif bottom_brightness > top_brightness * 1.1:
            return 'bottom'
    else:
        # Check left vs right
        left_quarter = gray[:, :w//4]
        right_quarter = gray[:, 3*w//4:]
        
        left_brightness = np.mean(left_quarter)
        right_brightness = np.mean(right_quarter)
        
        if left_brightness > right_brightness * 1.1:
            return 'left'
        elif right_brightness > left_brightness * 1.1:
            return 'right'
    
    return None


def predict_square_locations(
    strip_image: np.ndarray, 
    expected_count: int, 
    handle_end: Optional[str],
    is_vertical: bool
) -> List[Tuple[int, int, int]]:
    """Predict where squares should be based on strip dimensions."""
    h, w = strip_image.shape[:2]
    
    # Estimate square size (slightly smaller than strip width)
    square_size = int(w * 0.85)
    
    if is_vertical:
        # Squares arranged vertically
        # Skip handle area (assume 20% of height)
        handle_size = int(h * 0.2)
        
        if handle_end == 'top':
            start_y = handle_size
        else:
            start_y = 0
        
        # Calculate spacing
        available_height = h - handle_size
        total_square_height = square_size * expected_count
        spacing = (available_height - total_square_height) // (expected_count + 1) if expected_count > 1 else 0
        
        squares = []
        current_y = start_y + spacing
        for i in range(expected_count):
            x = (w - square_size) // 2  # Center horizontally
            squares.append((x, current_y, square_size))
            current_y += square_size + spacing
        
        return squares
    else:
        # Horizontal - similar logic
        handle_size = int(w * 0.2)
        
        if handle_end == 'left':
            start_x = handle_size
        else:
            start_x = 0
        
        available_width = w - handle_size
        total_square_width = square_size * expected_count
        spacing = (available_width - total_square_width) // (expected_count + 1) if expected_count > 1 else 0
        
        squares = []
        current_x = start_x + spacing
        for i in range(expected_count):
            y = (h - square_size) // 2  # Center vertically
            squares.append((current_x, y, square_size))
            current_x += square_size + spacing
        
        return squares


if __name__ == '__main__':
    import sys
    
    # Test images
    test_images = [
        'tests/fixtures/PXL_20250427_161114135.jpg',  # Vertical
        'tests/fixtures/PXL_20250427_161114135 copy.jpg',  # Horizontal
    ]
    
    if len(sys.argv) > 1:
        test_images = [sys.argv[1]]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            visualize_step_by_step(img_path, expected_pad_count=6)
        else:
            print(f"Image not found: {img_path}")

