"""
Debug utilities for PoolGuy CV Service.

Provides unified debugging interface with visual logging and step tracking.
"""

import logging
import numpy as np
import cv2
import inspect
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from utils.visual_logger import VisualLogger

logger = logging.getLogger(__name__)


@dataclass
class DebugStep:
    """Represents a single debug step in the pipeline."""
    step_id: str
    name: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None


class DebugContext:
    """
    Manages debug state and visual logging throughout the pipeline.
    
    Usage:
        debug = DebugContext(enabled=True, output_dir="logs")
        debug.add_step("01_strip", "Strip Detection", image, {"confidence": 0.95})
        debug.save_log()
    """
    
    def __init__(
        self,
        enabled: bool = False,
        output_dir: Optional[str] = None,
        image_name: str = "unknown",
        comparison_tag: Optional[str] = None,
        step_filter: Optional[List[str]] = None,
        track_parameters: bool = True
    ):
        """
        Initialize debug context.
        
        Args:
            enabled: Whether debug mode is enabled
            output_dir: Directory for saving visual logs
            image_name: Name of the image being processed
            comparison_tag: Optional tag for comparison mode (groups related runs)
            step_filter: Optional list of step IDs to log (None = log all)
            track_parameters: Whether to automatically track parameters used
        """
        self.enabled = enabled
        self.image_name = image_name
        self.comparison_tag = comparison_tag
        self.step_filter = step_filter
        self.track_parameters = track_parameters
        self.steps: List[DebugStep] = []
        self.visual_logger: Optional[VisualLogger] = None
        self.parameters_used: Dict[str, Any] = {}
        
        if enabled:
            self.visual_logger = VisualLogger(output_dir)
            if self.visual_logger:
                strategy_name = comparison_tag if comparison_tag else 'pipeline'
                self.visual_logger.start_log(strategy_name, image_name)
                # Store log directory for later retrieval (matches VisualLogger.save_log structure)
                from pathlib import Path
                image_base = Path(image_name).stem
                self._log_dir = Path(output_dir) / image_base / strategy_name if output_dir else None
            else:
                self._log_dir = None
        else:
            self._log_dir = None
    
    def add_step(
        self,
        step_id: str,
        name: str,
        image: Optional[Any] = None,
        data: Optional[Dict[str, Any]] = None,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a debug step.
        
        Args:
            step_id: Unique identifier for the step (e.g., "01_strip_detection")
            name: Human-readable step name
            image: Optional image to log (numpy array)
            data: Optional metadata dictionary
            description: Optional description of the step
            parameters: Optional parameters used in this step (auto-tracked if track_parameters=True)
        """
        if not self.enabled:
            return
        
        # Apply step filter
        if self.step_filter and step_id not in self.step_filter:
            return
        
        # Track parameters if enabled
        if self.track_parameters and parameters:
            self.parameters_used[step_id] = parameters
        
        # Capture caller's file and line number
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        source_info = {}
        if caller_frame:
            try:
                filename = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno
                # Get relative path from project root
                try:
                    from pathlib import Path
                    project_root = Path(__file__).parent.parent.parent
                    rel_path = Path(filename).relative_to(project_root)
                    source_info['source_file'] = str(rel_path)
                except (ValueError, AttributeError):
                    # If relative path fails, use absolute path
                    source_info['source_file'] = filename
                source_info['source_line'] = line_number
            except Exception:
                pass  # If inspection fails, continue without source info
        
        # Clean data to remove non-serializable types
        clean_data = {}
        if data:
            for key, value in data.items():
                clean_data[key] = self._clean_value(value)
        
        # Add source info to clean_data
        if source_info:
            clean_data['_source'] = source_info
        
        step = DebugStep(
            step_id=step_id,
            name=name,
            description=description,
            data=clean_data
        )
        self.steps.append(step)
        
        if self.visual_logger and image is not None:
            try:
                # Pass to visual logger: step_name, description, image, data
                # Use step_id as step_name, description as description, name as fallback description
                step_description = description if description else name
                self.visual_logger.add_step(step_id, step_description, image, clean_data)
            except Exception as e:
                logger.warning(f"Failed to add visual step {step_id}: {e}")
    
    def _clean_value(self, value):
        """Recursively clean a value for JSON serialization."""
        # Handle ImageTransformContext objects
        if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            # Check if it's an ImageTransformContext or similar object with to_dict method
            try:
                return self._clean_value(value.to_dict())
            except Exception:
                return f"<{type(value).__name__} object>"
        
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size < 100 else f"<ndarray shape={value.shape}>"
        elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8, np.bool_)):
            return int(value) if not isinstance(value, np.bool_) else bool(value)
        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (list, tuple)):
            return [self._clean_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        elif hasattr(value, '__dict__'):
            # Handle other objects by converting to dict
            try:
                return {k: self._clean_value(v) for k, v in value.__dict__.items()}
            except Exception:
                return f"<{type(value).__name__} object>"
        else:
            return value
    
    def save_log(self, final_image: Optional[Any] = None) -> Optional[str]:
        """
        Save the debug log.
        
        Args:
            final_image: Optional final image to save. If None, uses the last step's image.
            
        Returns:
            Path to saved log, or None if disabled
        """
        if not self.enabled or not self.visual_logger:
            return None
        
        try:
            # If no final image provided, try to get the last step's image from visual_logger
            if final_image is None:
                # Get the last step's image from visual_logger if available
                if self.visual_logger.steps and len(self.visual_logger.steps) > 0:
                    last_step = self.visual_logger.steps[-1]
                    if 'image' in last_step:
                        final_image = last_step['image']
                    else:
                        # Create a simple placeholder if no image available
                        logger.warning("No final image provided and no step images available")
                        return str(self._log_dir) if self._log_dir and self._log_dir.exists() else None
                else:
                    logger.warning("No steps available to use as final image")
                    return str(self._log_dir) if self._log_dir and self._log_dir.exists() else None
            
            log_path = self.visual_logger.save_log(final_image)
            return log_path if log_path else (str(self._log_dir) if self._log_dir and self._log_dir.exists() else None)
        except Exception as e:
            logger.warning(f"Failed to save debug log: {e}", exc_info=True)
            # Return stored log dir as fallback
            return str(self._log_dir) if self._log_dir and self._log_dir.exists() else None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get debug summary as dictionary.
        
        Returns:
            Dictionary with debug information
        """
        return {
            "enabled": self.enabled,
            "image_name": self.image_name,
            "comparison_tag": self.comparison_tag,
            "parameters_used": self.parameters_used,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "description": step.description,
                    "data": step.data,
                    "image_path": step.image_path
                }
                for step in self.steps
            ],
            "step_count": len(self.steps),
            "log_dir": str(self._log_dir) if self._log_dir else None
        }
    
    def is_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self.enabled
    
    def visualize_strip_detection(
        self,
        image: np.ndarray,
        strip_region: Dict[str, Any],
        method: str = "unknown",
        confidence: float = 0.0
    ) -> np.ndarray:
        """Create visualization for strip detection result."""
        vis = image.copy()
        cv2.rectangle(vis,
                     (strip_region['left'], strip_region['top']),
                     (strip_region['right'], strip_region['bottom']),
                     (0, 255, 0), 2)
        label = f"Strip: {method} ({confidence:.2f})"
        cv2.putText(vis, label,
                   (strip_region['left'], strip_region['top'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis
    
    def visualize_pad_detection(
        self,
        image: np.ndarray,
        pads: List[Dict[str, Any]],
        expected_count: Optional[int] = None
    ) -> np.ndarray:
        """Create visualization for pad detection result."""
        vis = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        
        if not pads:
            # Draw message if no pads detected
            cv2.putText(vis, 'No pads detected', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return vis
        
        h_img, w_img = vis.shape[:2]
        drawn_count = 0
        
        for i, pad in enumerate(pads):
            color = colors[i % len(colors)]
            # Handle PadRegion objects (has attributes) or dicts
            if hasattr(pad, 'x'):
                # PadRegion object
                x = pad.x
                y = pad.y
                w = pad.width
                h = pad.height
            else:
                # Dict format - handle both 'x'/'y' and 'left'/'top' formats
                x = pad.get('x', pad.get('left', 0))
                y = pad.get('y', pad.get('top', 0))
                w = pad.get('width', 0)
                h = pad.get('height', 0)
                # If width/height not available, calculate from right/bottom
                if w == 0 and 'right' in pad and 'left' in pad:
                    w = pad['right'] - pad['left']
                if h == 0 and 'bottom' in pad and 'top' in pad:
                    h = pad['bottom'] - pad['top']
            
            # Convert to int and ensure coordinates are within image bounds
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            
            # Skip if coordinates are invalid
            if w <= 0 or h <= 0:
                logger.warning(f"Pad {i+1} has invalid dimensions: {w}x{h}")
                continue
            
            # Clamp to image bounds
            x = max(0, min(w_img - 1, x))
            y = max(0, min(h_img - 1, y))
            w = max(1, min(w_img - x, w))
            h = max(1, min(h_img - y, h))
            
            # Draw pad rectangle with thicker line for visibility
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
            # Draw pad label
            label = f"P{i+1}"
            conf = pad.get('confidence', 0.0) if isinstance(pad, dict) else getattr(pad, 'confidence', 0.0)
            if conf > 0:
                label += f" {conf:.2f}"
            cv2.putText(vis, label, (x, max(15, y - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            drawn_count += 1
        
        # Add summary text
        pad_count = len(pads)
        summary = f"Detected: {pad_count} (drawn: {drawn_count})"
        if expected_count:
            summary += f" / Expected: {expected_count}"
        cv2.putText(vis, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if drawn_count == 0 and pad_count > 0:
            # Log warning if pads were provided but none could be drawn
            logger.warning(f"Provided {pad_count} pads but none could be drawn (invalid coordinates?)")
            cv2.putText(vis, f'WARNING: {pad_count} pads provided but none drawn', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis
    
    def visualize_final_result(
        self,
        image: np.ndarray,
        strip_region: Dict[str, Any],
        pads: List[Dict[str, Any]],
        overall_confidence: float = 0.0,
        use_actual_color: bool = True
    ) -> np.ndarray:
        """Create visualization for final pipeline result."""
        vis = image.copy()
        
        # Draw strip region
        cv2.rectangle(vis,
                     (strip_region['left'], strip_region['top']),
                     (strip_region['right'], strip_region['bottom']),
                     (0, 255, 0), 2)
        
        # Draw pads with colors
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 255), (0, 255, 255), (128, 0, 128)]
        for i, pad in enumerate(pads):
            region = pad.get('region', {})
            if region:
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('width', 0)
                h = region.get('height', 0)
                lab = pad.get('lab', {})
                color = colors[i % len(colors)]
                
                # Draw pad bounding box
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                label = f"P{i+1}: L={lab.get('L', 0):.0f}"
                cv2.putText(vis, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw color square to the right of the pad
                if lab and 'L' in lab and 'a' in lab and 'b' in lab:
                    if use_actual_color and region:
                        # Extract actual average color from the pad region in the image
                        pad_x = max(0, min(vis.shape[1] - 1, x))
                        pad_y = max(0, min(vis.shape[0] - 1, y))
                        pad_w = max(1, min(vis.shape[1] - pad_x, w))
                        pad_h = max(1, min(vis.shape[0] - pad_y, h))
                        pad_roi = vis[pad_y:pad_y + pad_h, pad_x:pad_x + pad_w]
                        if pad_roi.size > 0:
                            # Get mean BGR color from the actual pad region
                            bgr_color = tuple(map(int, np.mean(pad_roi.reshape(-1, 3), axis=0)))
                        else:
                            # Fallback to LAB conversion
                            bgr_color = (128, 128, 128)
                    else:
                        # Convert LAB to BGR for display
                        # Stored LAB values: L (0-100), a (-128 to 127), b (-128 to 127)
                        # OpenCV LAB format: L (0-100), a (0-255 with 128 neutral), b (0-255 with 128 neutral)
                        L_val = lab.get('L', 0)
                        a_val = lab.get('a', 0)
                        b_val = lab.get('b', 0)
                        
                        # Convert from stored format (-128 to 127) to OpenCV format (0-255)
                        # If values are small (likely in -128 to 127 range), add 128
                        # If values are already > 127, they're likely already in OpenCV format
                        # Check: if abs(a_val) < 128, it's in stored format, convert it
                        if abs(a_val) < 128:
                            a_val = a_val + 128
                        if abs(b_val) < 128:
                            b_val = b_val + 128
                        
                        # Clamp values to valid ranges
                        L_val = max(0, min(100, L_val))
                        a_val = max(0, min(255, a_val))
                        b_val = max(0, min(255, b_val))
                        
                        # Convert LAB to BGR
                        lab_array = np.uint8([[[int(L_val), int(a_val), int(b_val)]]])
                        bgr_color = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)[0][0]
                        bgr_color = tuple(map(int, bgr_color))
                    
                    # Calculate size of color square
                    h_img, w_img = vis.shape[:2]
                    square_size = min(max(h // 2, 20), 40)  # Smaller square that fits
                    
                    # Try different positions: right, left, or inside pad
                    square_x = None
                    square_y = None
                    
                    # Option 1: Right of pad
                    if x + w + 5 + square_size <= w_img:
                        square_x = x + w + 5
                        square_y = y
                    # Option 2: Left of pad
                    elif x - 5 - square_size >= 0:
                        square_x = x - 5 - square_size
                        square_y = y
                    # Option 3: Inside pad (top-right corner)
                    else:
                        square_x = x + w - square_size - 2
                        square_y = y + 2
                    
                    # Clamp to image bounds
                    square_x = max(0, min(w_img - square_size, square_x))
                    square_y = max(0, min(h_img - square_size, square_y))
                    
                    # Draw filled square with detected color
                    cv2.rectangle(vis,
                                 (square_x, square_y),
                                 (square_x + square_size, square_y + square_size),
                                 bgr_color, -1)  # -1 for filled
                    # Draw border around color square
                    cv2.rectangle(vis,
                                 (square_x, square_y),
                                 (square_x + square_size, square_y + square_size),
                                 (255, 255, 255), 2)  # White border
        
        cv2.putText(vis, f'Pads: {len(pads)}, Confidence: {overall_confidence:.2f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return vis
    
    def visualize_error(
        self,
        image: np.ndarray,
        error_message: str,
        error_code: Optional[str] = None
    ) -> np.ndarray:
        """Create visualization for error state."""
        vis = image.copy()
        text = error_message
        if error_code:
            text = f"{error_code}: {error_message}"
        cv2.putText(vis, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

