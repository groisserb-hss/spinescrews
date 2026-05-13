"""
Vendored articulated_models — spine articulation from humanfit.

Source: ~/dev/humanfit/humanfit/articulated_models/
Modified: pytools_bgroisser → bg3dtools.transforms_unified
"""

from .base_unified import Articulated
from .spine import Spine

__all__ = ['Articulated', 'Spine']
