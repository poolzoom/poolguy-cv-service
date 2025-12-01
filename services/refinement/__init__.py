"""
Strip refinement service for PoolGuy CV Service.

Refines YOLO-detected strip regions by:
- Normalizing orientation
- Tightening bounding box
- Removing excess background
- Preparing for warping
"""

from .strip_refiner import StripRefiner

__all__ = ['StripRefiner']








