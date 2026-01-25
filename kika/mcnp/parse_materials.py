"""Backwards-compatible re-export shim for material parsing.

This module re-exports material parsing functions from the new kika.materials
module to maintain backwards compatibility with existing code that imports from
kika.mcnp.parse_materials.

The parsing functions have been moved to kika.materials.parse_mcnp for better
organization. New code should import directly from kika.materials instead.

Examples
--------
>>> # Legacy import (still works)
>>> from kika.mcnp.parse_materials import read_material

>>> # Recommended new import
>>> from kika.materials import read_material
"""

from kika.materials.parse_mcnp import (
    read_material,
    _parse_nuclide_token,
    _parse_kika_name_comment,
    _parse_kika_density_comment,
)

__all__ = [
    'read_material',
    '_parse_nuclide_token',
    '_parse_kika_name_comment',
    '_parse_kika_density_comment',
]
