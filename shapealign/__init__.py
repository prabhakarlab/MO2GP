"""
shapealign package

Provides:
- ShapeAlign class for contour preprocessing, feature extraction, and embedding.
- Utility functions for contour handling and shape descriptor computation.
"""

from .align import ShapeAlign
from . import utils_contour
from . import utils_naive

__all__ = ["ShapeAlign", "utils_contour", "utils_naive"]
