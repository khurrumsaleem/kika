"""
EXFOR (Experimental Nuclear Reaction Data) utilities for KIKA.

This module provides classes and functions for loading, processing, and plotting
experimental angular distribution data from EXFOR.

Primary API:
    - read_exfor(): Load a single EXFOR JSON file
    - read_all_exfor(): Load all EXFOR files from a directory
    - ExforEntry: Base class for EXFOR entries
    - ExforAngularDistribution: Main class for angular distribution data

Example:
    >>> from kika.exfor import read_exfor
    >>> exfor = read_exfor('/path/to/27673002.json')
    >>> print(exfor.label)
    Gkatis et al. (2025)
    >>> print(exfor.energies())  # Returns numpy array of energies in MeV
    [1.0098, 1.0202, ...]

    # Convert to CM frame
    >>> exfor_cm = exfor.convert_to_cm()

    # Get data at specific energy
    >>> df = exfor_cm.to_dataframe(energy=1.5)

    # Get data in energy range with forward angles only
    >>> df = exfor_cm.to_dataframe(energy=(1.0, 2.0), angle=(0, 90))

Legacy API:
    The functions from AD_utils are still available but deprecated.
    Please migrate to the new class-based API.
"""

# Primary API - Main exports
from kika.exfor.io import read_exfor, read_all_exfor
from kika.exfor.exfor_entry import ExforEntry
from kika.exfor.angular_distribution import ExforAngularDistribution

# Transform functions
from kika.exfor.transforms import (
    cos_cm_from_cos_lab,
    cos_lab_from_cos_cm,
    jacobian_cm_to_lab,
    jacobian_lab_to_cm,
    transform_lab_to_cm,
    transform_cm_to_lab,
    angle_deg_to_cos,
    cos_to_angle_deg,
)

# Plotting functions
from kika.exfor.plotting import (
    plot_exfor_angular,
    plot_exfor_ace_comparison,
    plot_multiple_energies,
)

# Legacy API - Deprecated (imported for backward compatibility)
from kika.exfor.AD_utils import (
    # ACE Data Processing
    extract_angular_distribution,
    calculate_differential_cross_section,
    cosine_to_angle_degrees,
    angle_degrees_to_cosine,
    # EXFOR Data Processing (deprecated - use read_exfor instead)
    load_exfor_data,
    extract_experiment_info,
    load_all_exfor_data,
    # Plotting Functions (deprecated - use new plotting module)
    plot_angular_distribution,
    plot_combined_angular_distribution,
    plot_individual_energy_comparisons,
    plot_combined_angular_distribution_multi_ace,
    plot_all_energies_comparison,
)

__all__ = [
    # Primary API
    "read_exfor",
    "read_all_exfor",
    "ExforEntry",
    "ExforAngularDistribution",
    # Transforms
    "cos_cm_from_cos_lab",
    "cos_lab_from_cos_cm",
    "jacobian_cm_to_lab",
    "jacobian_lab_to_cm",
    "transform_lab_to_cm",
    "transform_cm_to_lab",
    "angle_deg_to_cos",
    "cos_to_angle_deg",
    # Plotting
    "plot_exfor_angular",
    "plot_exfor_ace_comparison",
    "plot_multiple_energies",
    # Legacy API (deprecated)
    "extract_angular_distribution",
    "calculate_differential_cross_section",
    "cosine_to_angle_degrees",
    "angle_degrees_to_cosine",
    "load_exfor_data",
    "extract_experiment_info",
    "load_all_exfor_data",
    "plot_angular_distribution",
    "plot_combined_angular_distribution",
    "plot_individual_energy_comparisons",
    "plot_combined_angular_distribution_multi_ace",
    "plot_all_energies_comparison",
]
