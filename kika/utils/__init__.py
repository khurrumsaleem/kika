"""Utility functions for KIKA."""
from kika.utils.logging_utils import configure_ace_debug_logging, configure_endf_debug_logging, get_endf_logger
from kika.utils.energy_folding import (
    EnergyFoldingConfig,
    FWHM_TO_SIGMA,
    compute_energy_resolution_tof,
    fold_cross_section,
    fold_angular_distribution,
    endf_angular_distribution,
    compute_folded_differential_xs,
    compute_unfolded_differential_xs,
)

__all__ = [
    # Logging utilities
    'configure_ace_debug_logging',
    'configure_endf_debug_logging',
    'get_endf_logger',
    # Energy folding utilities
    'EnergyFoldingConfig',
    'FWHM_TO_SIGMA',
    'compute_energy_resolution_tof',
    'fold_cross_section',
    'fold_angular_distribution',
    'endf_angular_distribution',
    'compute_folded_differential_xs',
    'compute_unfolded_differential_xs',
]
