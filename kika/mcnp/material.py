"""Backwards-compatible re-export shim for materials module.

This module re-exports material classes from the new kika.materials module
to maintain backwards compatibility with existing code that imports from
kika.mcnp.material.

The material classes have been moved to kika.materials for multi-code support.
New code should import directly from kika.materials instead.

Examples
--------
>>> # Legacy import (still works)
>>> from kika.mcnp.material import Material, MaterialCollection

>>> # Recommended new import
>>> from kika.materials import Material, MaterialCollection
"""

from kika.materials import Nuclide, NuclideAccessor, Material, MaterialCollection

# Re-export internal helpers for backwards compatibility
from kika.materials.material import (
    _DENSITY_UNIT_ALIASES,
    _DENSITY_CONVERSIONS,
    _normalize_density_unit,
    _validate_fraction_type,
    _is_natural_zaid,
)

__all__ = [
    'Nuclide',
    'NuclideAccessor',
    'Material',
    'MaterialCollection',
    '_DENSITY_UNIT_ALIASES',
    '_DENSITY_CONVERSIONS',
    '_normalize_density_unit',
    '_validate_fraction_type',
    '_is_natural_zaid',
]
