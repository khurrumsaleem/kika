"""
ENDF writers module for modifying and writing ENDF files.

This module provides utilities for:
- Modifying existing ENDF files (replace sections)
- Creating MF34 (angular distribution covariance) sections from scratch
- Writing covariance data to ENDF format
"""

from .endf_writer import ENDFWriter, replace_mf_section, replace_mt_section
from .mf34_writer import (
    create_mf34_from_covariance,
    write_mf34_to_file,
    remove_mf34_from_file,
    merge_mf34,
)

__all__ = [
    # ENDF file modification
    'ENDFWriter',
    'replace_mf_section',
    'replace_mt_section',
    # MF34 covariance creation and manipulation
    'create_mf34_from_covariance',
    'write_mf34_to_file',
    'remove_mf34_from_file',
    'merge_mf34',
]
