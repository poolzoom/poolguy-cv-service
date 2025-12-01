"""
Service interfaces and type definitions for PoolGuy CV Service.

This module defines all data structures used for communication between services,
ensuring type safety and clear contracts.
"""

from typing import TypedDict, List, Optional, Literal


class StripRegion(TypedDict):
    """
    Strip region coordinates and metadata.
    
    All coordinates are absolute (relative to original image).
    """
    left: int
    top: int
    right: int
    bottom: int
    width: int
    height: int
    confidence: float
    detection_method: Literal["yolo", "opencv", "openai", "yolo_pca", "yolo_refined"]
    orientation: Literal["vertical", "horizontal"]
    angle: float  # Rotation angle in degrees


class PadRegion(TypedDict):
    """
    Pad region coordinates.
    
    All coordinates are absolute (relative to original image).
    Provides both (x, y, width, height) and (left, top, right, bottom) formats.
    """
    pad_index: int
    x: int  # Left coordinate (same as left)
    y: int  # Top coordinate (same as top)
    width: int
    height: int
    left: int  # Left coordinate
    top: int  # Top coordinate
    right: int  # Right coordinate (x + width)
    bottom: int  # Bottom coordinate (y + height)


class LabColor(TypedDict):
    """LAB color space values."""
    L: float  # Lightness (0-100)
    a: float  # Green-Red axis (-128 to 127)
    b: float  # Blue-Yellow axis (-128 to 127)


class ColorResult(TypedDict):
    """
    Color extraction result for a single pad.
    
    Includes pad coordinates, LAB color values, and confidence metrics.
    """
    pad_index: int
    region: PadRegion  # Pad coordinates
    lab: LabColor  # LAB color values
    confidence: float  # Detection confidence (0.0-1.0)
    color_variance: float  # Color variance within pad
    pad_detection_confidence: float  # Pad detection confidence


class DetectionResult(TypedDict):
    """
    Complete detection result from strip detection service.
    
    Includes strip region and optional pad regions if detected.
    """
    success: bool
    strip: Optional[StripRegion]  # Strip region if detected
    error: Optional[str]  # Error message if failed
    error_code: Optional[str]  # Error code if failed
    visual_log_path: Optional[str]  # Path to visual log if debug enabled


class PadDetectionResult(TypedDict):
    """
    Pad detection result.
    
    Contains list of detected pad regions with coordinates.
    """
    success: bool
    pads: List[PadRegion]  # List of detected pad regions
    error: Optional[str]  # Error message if failed
    error_code: Optional[str]  # Error code if failed
    detected_count: int  # Number of pads detected


class ColorExtractionResult(TypedDict):
    """
    Color extraction result.
    
    Contains color data for all detected pads.
    """
    success: bool
    pads: List[ColorResult]  # List of pad color results
    overall_confidence: float  # Average confidence across all pads
    error: Optional[str]  # Error message if failed
    error_code: Optional[str]  # Error code if failed


class FullPipelineResult(TypedDict):
    """
    Complete pipeline result.
    
    Includes strip detection, pad detection, and color extraction results.
    """
    success: bool
    strip: StripRegion
    pads: List[ColorResult]  # Pads with colors
    overall_confidence: float
    processing_time_ms: int
    debug: Optional[dict]  # Debug information if debug mode enabled


class ReferenceSquare(TypedDict):
    """
    Reference color square for bottle test strips.
    
    Contains color information and associated value/range.
    """
    color: LabColor  # LAB color of the reference square
    value: Optional[str]  # Associated value (e.g., "7.2", "low", "high")
    region: PadRegion  # Coordinates of the reference square
    confidence: float  # Detection confidence (0.0-1.0)
    associated_pad: Optional[int]  # Optional pad index this square is associated with


class BottlePadRegion(TypedDict):
    """
    Pad region on a test strip bottle with name and reference information.
    
    All coordinates are absolute (relative to original image).
    """
    pad_index: int
    name: Optional[str]  # Pad name from OCR (e.g., "pH", "Chlorine")
    region: PadRegion  # Pad coordinates
    reference_range: Optional[str]  # Reference range text (e.g., "7.2-7.8")
    reference_squares: List[ReferenceSquare]  # Reference color squares
    detected_color: Optional[LabColor]  # Detected color from pad
    mapped_value: Optional[str]  # Mapped value based on reference squares
    confidence: float  # Overall confidence (0.0-1.0)


class BottlePipelineResult(TypedDict):
    """
    Complete bottle pipeline result.
    
    Includes pad names, reference ranges, and color mappings.
    """
    success: bool
    pads: List[BottlePadRegion]  # List of detected pads with names and colors
    overall_confidence: float  # Average confidence across all pads
    images_processed: int  # Number of images processed
    pads_detected: int  # Total number of pads detected
    error: Optional[str]  # Error message if failed
    error_code: Optional[str]  # Error code if failed

