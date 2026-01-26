"""
Utility functions for EXFOR-to-ENDF angular distribution sampling.

This module contains reusable functions for:
- EXFOR data loading and filtering
- Kernel weighting and diagnostics
- Energy binning with TOF resolution
- Covariance computation
- ENDF file writing
- MF34 covariance generation

These functions are used by the main workflow in exfor_to_endf_sampling.py.
"""
from __future__ import annotations

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

# Add kika to path if needed
_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# Import kika modules
from kika.endf.read_endf import read_endf
from kika.endf.writers.endf_writer import ENDFWriter
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed

# Use new kika.exfor module for transforms
from kika.exfor.transforms import transform_lab_to_cm, jacobian_cm_to_lab
from kika.exfor.angular_distribution import ExforAngularDistribution

# Import resample_AD functions (relative import for same directory)
from .resample_AD import (
    load_exfor_for_fitting,
    endf_normalize_legendre_coeffs,
    compute_energy_resolution_tof,
    compute_n_eff,
    compute_weight_span_95,
)


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

class DualLogger:
    """Logger that writes to both file and optionally to console."""

    def __init__(self, log_file: str):
        self.log_file = log_file

        # Create logger
        self.logger = logging.getLogger('exfor_to_endf')
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def info(self, msg: str, console: bool = False):
        self.logger.info(msg)
        if console:
            print(f"[INFO] {msg}")

    def warning(self, msg: str, console: bool = False):
        self.logger.warning(f"[WARNING] {msg}")
        if console:
            print(f"[WARNING] {msg}")

    def error(self, msg: str, console: bool = False):
        self.logger.error(f"[ERROR] {msg}")
        if console:
            print(f"[ERROR] {msg}")

    def debug(self, msg: str):
        self.logger.debug(msg)


# Global logger instance
_logger: Optional[DualLogger] = None


def _get_logger() -> Optional[DualLogger]:
    """Get the global logger instance."""
    return _logger


def _set_logger(logger: Optional[DualLogger]) -> None:
    """Set the global logger instance."""
    global _logger
    _logger = logger


def _format_condensed_experiments(experiments_info: List[Dict]) -> List[str]:
    """
    Group experiments by (entry, subentry) and format as condensed log lines.

    Instead of listing each experiment occurrence separately, this groups
    multiple occurrences of the same experiment and summarizes:
    - Number of energies used (and if deduplication occurred)
    - Energy range
    - Total number of angular points
    - Weight range

    Parameters
    ----------
    experiments_info : List[Dict]
        List of experiment info dicts with keys:
        entry, subentry, author, year, exfor_energy_mev, kernel_weight, n_points
        Optionally includes 'selected_from_n_energies' for deduplication info.

    Returns
    -------
    List[str]
        Formatted log lines, one per unique experiment
    """
    if not experiments_info:
        return []

    # Group by (entry, subentry)
    grouped = defaultdict(lambda: {
        'author': '',
        'year': '',
        'energies': [],
        'weights': [],
        'total_points': 0,
        'selected_from_n_energies': 0,  # Track deduplication
    })

    for exp in experiments_info:
        key = (exp['entry'], exp['subentry'])
        grouped[key]['author'] = exp['author']
        grouped[key]['year'] = exp['year']
        grouped[key]['energies'].append(exp['exfor_energy_mev'])
        grouped[key]['weights'].append(exp['kernel_weight'])
        grouped[key]['total_points'] += exp['n_points']
        # Track deduplication info (use max in case there are multiple)
        n_in_bin = exp.get('selected_from_n_energies', 1)
        if n_in_bin > grouped[key]['selected_from_n_energies']:
            grouped[key]['selected_from_n_energies'] = n_in_bin

    # Format each group
    lines = []
    for (entry, subentry), data in sorted(grouped.items()):
        exp_id = f"{entry}.{subentry}"
        author = data['author']
        year = data['year']

        energies = sorted(data['energies'])
        n_energies = len(energies)
        e_min, e_max = energies[0], energies[-1]

        weights = data['weights']
        w_min, w_max = min(weights), max(weights)

        total_pts = data['total_points']

        # Track deduplication
        n_in_bin = data['selected_from_n_energies']

        # Format energy string with deduplication info
        if n_energies == 1:
            if n_in_bin > 1:
                # Single energy selected from multiple in bin
                energy_str = f"{e_min:.4f} MeV (closest of {n_in_bin} in bin)"
            else:
                energy_str = f"{e_min:.4f} MeV"
        else:
            # Multiple energies used (shouldn't happen with deduplication, but handle anyway)
            if n_in_bin > n_energies:
                energy_str = f"{n_energies} energies [{e_min:.4f}-{e_max:.4f} MeV] (from {n_in_bin} in bin)"
            else:
                energy_str = f"{n_energies} energies [{e_min:.4f}-{e_max:.4f} MeV]"

        # Format weight string
        if abs(w_max - w_min) < 0.001:
            weight_str = f"w={w_min:.3f}"
        else:
            weight_str = f"w=[{w_min:.3f}-{w_max:.3f}]"

        line = f"  - {exp_id} ({author}, {year}): {energy_str}, {total_pts} pts, {weight_str}"
        lines.append(line)

    return lines


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnergyBinInfo:
    """Information about an energy bin for EXFOR fitting."""
    index: int                           # Index in the energy grid
    energy_ev: float                     # Energy in eV
    energy_mev: float                    # Energy in MeV
    sigma_E_mev: float = 0.0             # Energy resolution from TOF (MeV)
    bin_lower_mev: float = 0.0           # Lower bin boundary (MeV) for energy_bin method
    bin_upper_mev: float = float('inf')  # Upper bin boundary (MeV) for energy_bin method
    original_coeffs: List[float] = field(default_factory=list)  # Original Legendre coefficients
    has_exfor_data: bool = False         # Whether EXFOR data was found
    exfor_n_points: int = 0              # Number of EXFOR data points
    exfor_n_experiments: int = 0         # Number of matching experiments
    experiments_used: List[Dict] = field(default_factory=list)  # List of experiments used
    fitted_degree: int = 0               # Fitted Legendre degree
    chi2_red: float = 0.0                # Reduced chi-squared of fit
    tau_F: float = 0.0                   # Forward band discrepancy
    tau_M: float = 0.0                   # Mid band discrepancy
    tau_B: float = 0.0                   # Backward band discrepancy
    interpolated: bool = False           # Whether coefficients were interpolated


@dataclass
class SamplingResult:
    """Result from sampling at one energy bin."""
    bin_info: EnergyBinInfo
    sampled_coeffs: Optional[np.ndarray] = None  # Shape: (n_samples, n_coeffs) - ENDF format (a_1, a_2, ...)
    fit_info: Optional[Dict[str, Any]] = None


@dataclass
class KernelDiagnostics:
    """Diagnostics for kernel weighting at one energy point.

    These metrics help assess the quality of the Gaussian kernel weighting:
    - n_eff: Effective sample size (higher is better)
    - weight_span_95: Energy interval containing 95% of weight
    - weight_span_ratio: weight_span_95 / σE (should be close to 2-3)
    - max_experiment_weight_frac: Largest single experiment contribution
    - capping_applied: Whether any experiment was weight-capped
    """
    n_eff: float                                    # Effective sample size
    weight_span_95: float                           # 95% weight span in MeV
    weight_span_ratio: float                        # weight_span_95 / sigma_E
    n_experiments: int                              # Number of experiments contributing
    max_experiment_weight_frac: float               # Largest single experiment weight fraction
    experiment_weights: Dict[str, float]            # {exp_key: weight_frac} before capping
    n_points_dropped: int                           # Points dropped by min weight threshold
    capping_applied: bool                           # Whether experiment capping was applied


# =============================================================================
# RESOLUTION OVERLAP WEIGHTING
# =============================================================================

def compute_overlap_weight(
    exp_energy_mev: float,
    sigma_E_mev: float,
    bin_lower_mev: float,
    bin_upper_mev: float,
) -> float:
    """
    Compute probability that measurement's true energy lies within bin.

    Instead of Gaussian distance-based weighting, this computes the probability
    that an experiment's true energy falls within the ENDF bin, given the
    experiment's energy resolution.

    Formula:
        w = Φ((E_high - E_j)/σ_j) - Φ((E_low - E_j)/σ_j)

    Where:
        Φ = standard normal CDF
        E_j = experimental measurement energy
        σ_j = experiment-specific energy resolution
        [E_low, E_high] = ENDF bin boundaries

    Properties:
        - E inside bin + good resolution → w ≈ 1
        - E outside bin + good resolution → w ≈ 0 (no dragging in off-energy data)
        - Poor resolution → smears across bins (physics-consistent)

    Parameters
    ----------
    exp_energy_mev : float
        Experimental measurement energy in MeV
    sigma_E_mev : float
        Experiment-specific energy resolution in MeV
    bin_lower_mev : float
        Lower boundary of ENDF bin in MeV
    bin_upper_mev : float
        Upper boundary of ENDF bin in MeV

    Returns
    -------
    float
        Probability weight [0, 1] that the true energy is in the bin
    """
    if sigma_E_mev <= 0:
        # Perfect resolution: 1 if inside bin, 0 otherwise
        return 1.0 if bin_lower_mev <= exp_energy_mev <= bin_upper_mev else 0.0

    z_high = (bin_upper_mev - exp_energy_mev) / sigma_E_mev
    z_low = (bin_lower_mev - exp_energy_mev) / sigma_E_mev

    return norm.cdf(z_high) - norm.cdf(z_low)


# =============================================================================
# ENERGY BINNING WITH TOF RESOLUTION
# =============================================================================

def compute_energy_bins_with_tof_resolution(
    energies_ev: np.ndarray,
    energy_min_mev: float,
    energy_max_mev: float,
    delta_t_ns: float = 10.0,
    flight_path_m: float = 27.037,
) -> List[EnergyBinInfo]:
    """
    Compute energy bins with TOF-based energy resolution.

    Uses TOF (Time-of-Flight) parameters to compute energy resolution σE(E)
    at each energy point, which determines the Gaussian kernel width for
    including experimental data.

    Also computes bin boundaries for the energy_bin selection method:
    - Lower bound: midpoint to previous energy point
    - Upper bound: midpoint to next energy point

    Parameters
    ----------
    energies_ev : np.ndarray
        Energy grid in eV
    energy_min_mev : float
        Minimum energy to process (in MeV)
    energy_max_mev : float
        Maximum energy to process (in MeV)
    delta_t_ns : float
        TOF time resolution in nanoseconds
    flight_path_m : float
        TOF flight path in meters

    Returns
    -------
    List[EnergyBinInfo]
        List of energy bin info objects with computed σE and bin boundaries
    """
    logger = _get_logger()

    energies_mev = energies_ev / 1e6  # Convert to MeV

    # First pass: identify indices in range
    indices_in_range = []
    for i, e_mev in enumerate(energies_mev):
        if e_mev >= energy_min_mev and e_mev <= energy_max_mev:
            indices_in_range.append(i)

    bins = []
    n_bins = len(indices_in_range)

    for local_idx, global_idx in enumerate(indices_in_range):
        e_mev = energies_mev[global_idx]

        # Compute TOF-based energy resolution
        sigma_E = compute_energy_resolution_tof(
            E_mev=e_mev,
            delta_t_ns=delta_t_ns,
            flight_path_m=flight_path_m,
        )

        # Compute bin boundaries (midpoints to neighbors)
        # Lower boundary
        if local_idx == 0:
            # First bin in range: use midpoint to previous grid point if available
            if global_idx > 0:
                bin_lower = (energies_mev[global_idx - 1] + e_mev) / 2.0
            else:
                bin_lower = 0.0  # No previous point, extend down to 0
        else:
            prev_global_idx = indices_in_range[local_idx - 1]
            bin_lower = (energies_mev[prev_global_idx] + e_mev) / 2.0

        # Upper boundary
        if local_idx == n_bins - 1:
            # Last bin in range: use midpoint to next grid point if available
            if global_idx < len(energies_mev) - 1:
                bin_upper = (e_mev + energies_mev[global_idx + 1]) / 2.0
            else:
                bin_upper = float('inf')  # No next point, extend up
        else:
            next_global_idx = indices_in_range[local_idx + 1]
            bin_upper = (e_mev + energies_mev[next_global_idx]) / 2.0

        bin_info = EnergyBinInfo(
            index=global_idx,
            energy_ev=energies_ev[global_idx],
            energy_mev=e_mev,
            sigma_E_mev=sigma_E,
            bin_lower_mev=bin_lower,
            bin_upper_mev=bin_upper,
        )
        bins.append(bin_info)

    if logger:
        logger.info(f"Computed tolerances for {len(bins)} energy bins in range [{energy_min_mev:.3f}, {energy_max_mev:.3f}] MeV")

    return bins


# =============================================================================
# EXFOR CACHE BUILDING
# =============================================================================

def build_exfor_cache_from_objects(
    exfor_objects: List[ExforAngularDistribution],
    exclude_experiments: Optional[List[str]] = None,
) -> Tuple[Dict[float, List[Tuple[pd.DataFrame, Dict]]], List[float]]:
    """
    Build an EXFOR data cache from ExforAngularDistribution objects.

    This function converts a list of ExforAngularDistribution objects into the
    cache format expected by filter_exfor_with_kernel_weights. For each object,
    it uses to_dataframe() to extract data at each available energy.

    Parameters
    ----------
    exfor_objects : List[ExforAngularDistribution]
        List of EXFOR angular distribution objects to cache
    exclude_experiments : List[str], optional
        List of experiments to exclude from the cache. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above

    Returns
    -------
    Tuple[Dict[float, List[Tuple[pd.DataFrame, Dict]]], List[float]]
        - exfor_cache: Dict mapping energy (MeV) to list of (DataFrame, metadata) tuples
        - sorted_energies: Sorted list of all available energies (MeV)

    Notes
    -----
    The returned DataFrame has columns compatible with filter_exfor_with_kernel_weights:
        - 'angle': Angle in degrees
        - 'dsig': Differential cross section in b/sr
        - 'error_stat': Statistical uncertainty in b/sr

    The metadata dict contains:
        - 'entry': EXFOR entry number
        - 'subentry': EXFOR subentry number
        - 'angle_frame': Reference frame ('CM' or 'LAB')
        - 'reaction': Reaction notation
        - 'citation': Citation dict with authors, year, etc.
        - 'energy_resolution_inputs': TOF parameters if available
    """
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]] = {}
    all_energies = set()

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    for exfor in exfor_objects:
        # Check if experiment is excluded
        if _is_experiment_excluded(exfor.entry, exfor.subentry, exclusion_patterns):
            continue

        # Get all available energies in MeV
        energies_mev = exfor.energies(unit='MeV')

        for energy_mev in energies_mev:
            # Get data at this energy using to_dataframe
            # Use small tolerance (0.1 keV) to avoid mixing data from different energies
            # while allowing for floating point precision issues
            df = exfor.to_dataframe(
                energy=energy_mev,
                energy_unit='MeV',
                cross_section_unit='b/sr',
                angle_unit='deg',
                tolerance=1e-4,  # 0.1 keV tolerance
            )

            if df.empty:
                continue

            # Convert DataFrame to expected column names
            cache_df = pd.DataFrame({
                'angle': df['angle'].values,
                'dsig': df['value'].values,
                'error_stat': df['error'].values,
            })

            # Build metadata dict
            meta = {
                'entry': exfor.entry,
                'subentry': exfor.subentry,
                'angle_frame': exfor.angle_frame,
                'reaction': exfor.reaction.get('notation', ''),
                'citation': exfor.citation,
            }

            # Add energy resolution inputs if available
            energy_res = exfor.method.get('energy_resolution_input') or exfor.method.get('energy_resolution_inputs')
            if energy_res:
                distance = energy_res.get('distance', {})
                time_res = energy_res.get('time_resolution', {})
                meta['energy_resolution_inputs'] = {
                    'flight_path_m': distance.get('value'),
                    'time_resolution_ns': time_res.get('value'),
                }

            # Add to cache
            if energy_mev not in exfor_cache:
                exfor_cache[energy_mev] = []
            exfor_cache[energy_mev].append((cache_df, meta))
            all_energies.add(energy_mev)

    sorted_energies = sorted(all_energies)
    return exfor_cache, sorted_energies


# =============================================================================
# EXFOR FILTERING FUNCTIONS
# =============================================================================


def _parse_exclusion_list(exclude_list: Optional[List[str]]) -> set:
    """
    Parse exclusion list into a set of normalized patterns for matching.

    Accepts multiple formats:
    - "20743" - excludes all subentries starting with 20743
    - "20743002" - excludes specific dataset
    - "20743/002" - same as above

    Parameters
    ----------
    exclude_list : List[str], optional
        List of experiment IDs to exclude

    Returns
    -------
    set
        Set of (entry_prefix, full_id) tuples for matching.
        entry_prefix is for matching all subentries, full_id for exact match.
    """
    if not exclude_list:
        return set()

    patterns = set()
    for item in exclude_list:
        item = item.strip()
        if not item:
            continue

        # Handle "entry/subentry" format
        if "/" in item:
            parts = item.split("/")
            entry = parts[0].strip()
            subentry = parts[1].strip() if len(parts) > 1 else ""
            full_id = entry + subentry
            patterns.add(full_id)
        elif len(item) <= 5:
            # Short ID - treat as entry prefix (matches all subentries)
            patterns.add(("prefix", item))
        else:
            # Full dataset ID
            patterns.add(item)

    return patterns


def _is_experiment_excluded(
    entry: str,
    subentry: str,
    exclusion_patterns: set,
) -> bool:
    """
    Check if an experiment matches any exclusion pattern.

    Parameters
    ----------
    entry : str
        EXFOR entry number (e.g., "20743")
    subentry : str
        EXFOR subentry number (e.g., "002")
    exclusion_patterns : set
        Set of patterns from _parse_exclusion_list()

    Returns
    -------
    bool
        True if experiment should be excluded
    """
    if not exclusion_patterns:
        return False

    # Build full dataset ID
    full_id = entry + subentry

    for pattern in exclusion_patterns:
        if isinstance(pattern, tuple) and pattern[0] == "prefix":
            # Prefix match - exclude all subentries of this entry
            if full_id.startswith(pattern[1]):
                return True
        elif full_id == pattern:
            # Exact match
            return True

    return False


def filter_exfor_with_kernel_weights(
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    energy_mev: float,
    sigma_E_mev: float,
    n_sigma: float,
    m_proj_u: float,
    m_targ_u: float,
    bin_lower_mev: float = 0.0,
    bin_upper_mev: float = float('inf'),
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    default_delta_t_ns: float = 10.0,
    default_flight_path_m: float = 27.037,
    use_overlap_weights: bool = True,
    normalize_by_n_points: bool = True,
    dedupe_per_experiment: bool = True,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    logger = None,
) -> Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]:
    """
    Filter EXFOR data using resolution-aware kernel weighting with diagnostics.

    Two weighting modes are available:

    1. Overlap weights (use_overlap_weights=True, recommended):
       w = Φ((E_high - E_j)/σ_j) - Φ((E_low - E_j)/σ_j)
       Computes probability that the true energy lies within the bin.

    2. Gaussian kernel (use_overlap_weights=False, legacy):
       g_ij = exp(-0.5 * ((E_i - E_j)/σE_j)²)
       Distance-based weighting.

    Per-energy normalization (normalize_by_n_points=True):
       Divides weight by number of angular points at each energy
       to prevent experiments with many angles from dominating.

    Per-experiment deduplication (dedupe_per_experiment=True):
       If an experiment has multiple energies within the kernel range,
       only the energy with the highest kernel weight is selected.
       This prevents experiments with dense energy sampling from dominating.

    Parameters
    ----------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Pre-loaded EXFOR data organized by energy
    sorted_energies : List[float]
        Sorted list of available energies in cache
    energy_mev : float
        Target ENDF grid energy in MeV
    sigma_E_mev : float
        Default energy resolution at target energy in MeV (fallback)
    n_sigma : float
        Cutoff in units of σE (typically 3.0)
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
    bin_lower_mev : float
        Lower boundary of ENDF bin in MeV (for overlap weights)
    bin_upper_mev : float
        Upper boundary of ENDF bin in MeV (for overlap weights)
    min_kernel_weight_fraction : float
        Minimum kernel weight as fraction of max (default: 1e-3)
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per experiment (default: 0.5)
    default_delta_t_ns : float
        Default time resolution in nanoseconds for fallback (default: 10.0)
    default_flight_path_m : float
        Default flight path in meters for fallback (default: 27.037)
    use_overlap_weights : bool
        If True, use resolution overlap weighting (recommended).
        If False, use legacy Gaussian kernel weighting.
    normalize_by_n_points : bool
        If True, divide weight by number of angular points at each energy.
        This prevents experiments with dense angular sampling from dominating.
    dedupe_per_experiment : bool
        If True (default), select only the highest-weighted energy for each
        experiment. This prevents experiments with many energies in range from
        dominating the fit.
    exclude_experiments : List[str], optional
        List of experiments to exclude from filtering. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above
    min_relative_uncertainty : float, optional
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty. This prevents experiments
        with unrealistically small uncertainties from dominating the fit.
    logger : logging.Logger, optional
        Logger for reporting fallback usage

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]
        - DataFrame with EXFOR data including 'kernel_weight' column
        - List of experiment metadata dicts (includes 'selected_from_n_energies')
        - Array of kernel weights for each data point
        - KernelDiagnostics object with N_eff, weight span, etc.
    """
    # Empty diagnostics for early returns
    empty_diag = KernelDiagnostics(
        n_eff=0.0, weight_span_95=0.0, weight_span_ratio=0.0,
        n_experiments=0, max_experiment_weight_frac=0.0,
        experiment_weights={}, n_points_dropped=0, capping_applied=False
    )

    if not exfor_cache or not sorted_energies:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    all_frames = []
    experiments_info = []
    all_kernel_weights = []

    # Track experiments that used fallback (log once per experiment, not per energy)
    fallback_logged = set()

    # Step 1: Collect all candidate data with computed kernel weights, grouped by experiment
    # experiment_candidates: {(entry, subentry): [(energy, df, meta, kernel_weight, exp_sigma_E, used_fallback), ...]}
    experiment_candidates: Dict[Tuple[str, str], List[Tuple[float, pd.DataFrame, Dict, float, float, bool]]] = defaultdict(list)

    for available_energy in sorted_energies:
        entries = exfor_cache.get(available_energy, [])
        for df, meta in entries:
            # Extract metadata
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')

            # Check if experiment is excluded
            if _is_experiment_excluded(entry, subentry, exclusion_patterns):
                continue

            # Get experiment-specific TOF parameters for energy resolution
            energy_res = meta.get('energy_resolution_inputs')

            if (energy_res and
                energy_res.get('flight_path_m') is not None and
                energy_res.get('time_resolution_ns') is not None):
                # Compute experiment-specific sigma_E at the MEASUREMENT energy
                # This is important: use available_energy, not energy_mev
                exp_sigma_E = compute_energy_resolution_tof(
                    E_mev=available_energy,  # Use measurement energy
                    delta_t_ns=energy_res['time_resolution_ns'],
                    flight_path_m=energy_res['flight_path_m'],
                )
                used_fallback = False
            else:
                # Fallback to default parameters at measurement energy
                exp_sigma_E = compute_energy_resolution_tof(
                    E_mev=available_energy,  # Use measurement energy
                    delta_t_ns=default_delta_t_ns,
                    flight_path_m=default_flight_path_m,
                )
                used_fallback = True

            # Compute weight based on method
            if use_overlap_weights:
                # Resolution overlap: probability that true energy is in bin
                kernel_weight = compute_overlap_weight(
                    exp_energy_mev=available_energy,
                    sigma_E_mev=exp_sigma_E,
                    bin_lower_mev=bin_lower_mev,
                    bin_upper_mev=bin_upper_mev,
                )
                # Skip if weight is negligible
                if kernel_weight < min_kernel_weight_fraction:
                    continue
            else:
                # Legacy Gaussian kernel weighting
                # Check cutoff using THIS experiment's sigma_E
                exp_cutoff = n_sigma * exp_sigma_E
                if available_energy < (energy_mev - exp_cutoff) or available_energy > (energy_mev + exp_cutoff):
                    continue
                # Compute Gaussian kernel weight
                delta_E = abs(available_energy - energy_mev)
                kernel_weight = np.exp(-0.5 * (delta_E / exp_sigma_E)**2)

            exp_key = (entry, subentry)
            experiment_candidates[exp_key].append((available_energy, df, meta, kernel_weight, exp_sigma_E, used_fallback))

    # Step 2: For each experiment, select highest-weighted energy (or all if dedupe disabled)
    selected_data: List[Tuple[float, pd.DataFrame, Dict, float, float, bool]] = []

    for exp_key, candidates in experiment_candidates.items():
        if dedupe_per_experiment and len(candidates) > 1:
            # Select the energy with highest kernel weight
            best = max(candidates, key=lambda x: x[3])  # x[3] = kernel_weight
            selected_data.append(best)
        else:
            selected_data.extend(candidates)

    # Step 3: Process selected data (transform and build DataFrames)
    for available_energy, df, meta, kernel_weight, exp_sigma_E, used_fallback in selected_data:
        # Extract metadata
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        # Log fallback only once per experiment (subentry)
        if used_fallback and logger and subentry not in fallback_logged:
            logger.info(f"  Using default TOF params for {subentry} "
                       f"(delta_t={default_delta_t_ns}ns, L={default_flight_path_m}m)")
            fallback_logged.add(subentry)

        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Extract columns
        angles_deg = df['angle'].to_numpy(dtype=float)
        dsig = df['dsig'].to_numpy(dtype=float)
        error_stat = df['error_stat'].to_numpy(dtype=float)

        n_points = len(angles_deg)

        # Count how many energies this experiment had in range
        n_energies_in_range = len(experiment_candidates[(entry, subentry)])

        # Per-energy normalization (Upgrade 2)
        # Divide weight by number of angular points to prevent
        # experiments with dense angular sampling from dominating
        if normalize_by_n_points and n_points > 0:
            point_weight = kernel_weight / n_points
        else:
            point_weight = kernel_weight

        # Transform to CM frame if needed
        if frame == 'LAB':
            mu_lab = np.cos(np.deg2rad(angles_deg))
            mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                mu_lab, dsig, error_stat, m_proj_u, m_targ_u
            )

            angles_cm_deg = np.rad2deg(np.arccos(mu_cm))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'frame': 'CM',
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': point_weight,  # Use normalized point weight
            })
        else:
            mu_cm = np.cos(np.deg2rad(angles_deg))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_deg,
                'value': dsig,
                'unc': error_stat,
                'mu': mu_cm,
                'frame': frame,
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': point_weight,  # Use normalized point weight
            })

        all_frames.append(transformed_df)
        all_kernel_weights.extend([point_weight] * n_points)

        # Track experiment info (store original kernel_weight for diagnostics)
        exp_info = {
            'entry': entry,
            'subentry': subentry,
            'author': author,
            'year': year,
            'exfor_energy_mev': available_energy,
            'kernel_weight': kernel_weight,  # Original weight before normalization
            'point_weight': point_weight,    # Weight after per-energy normalization
            'n_points': n_points,
            'sigma_E_mev': exp_sigma_E,
            'used_fallback_tof': used_fallback,
            'selected_from_n_energies': n_energies_in_range,  # NEW: track deduplication
        }
        experiments_info.append(exp_info)

    if not all_frames:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)
    kernel_weights = np.array(all_kernel_weights)

    # Apply uncertainty floor if requested
    if min_relative_uncertainty > 0:
        result = apply_uncertainty_floor(result, min_relative_uncertainty, unc_column='unc', value_column='value')

    # Apply minimum weight threshold
    kernel_weights, n_dropped = apply_min_weight_threshold(
        kernel_weights, min_kernel_weight_fraction
    )

    # Apply per-experiment weight capping
    kernel_weights, exp_weight_fracs, capping_applied = apply_per_experiment_weight_cap(
        result, kernel_weights, max_experiment_weight_fraction
    )

    # Update kernel_weight column in DataFrame
    result['kernel_weight'] = kernel_weights

    # Compute diagnostics
    exfor_energies = result['exfor_energy_mev'].values
    weight_span_95 = compute_weight_span_95(kernel_weights, exfor_energies, energy_mev)
    n_eff_prelim = compute_n_eff(kernel_weights, np.ones_like(kernel_weights))
    max_exp_frac = max(exp_weight_fracs.values()) if exp_weight_fracs else 0.0

    diagnostics = KernelDiagnostics(
        n_eff=n_eff_prelim,
        weight_span_95=weight_span_95,
        weight_span_ratio=weight_span_95 / sigma_E_mev if sigma_E_mev > 0 else 0.0,
        n_experiments=len(set(exp_weight_fracs.keys())) if exp_weight_fracs else len(experiments_info),
        max_experiment_weight_frac=max_exp_frac,
        experiment_weights=exp_weight_fracs,
        n_points_dropped=n_dropped,
        capping_applied=capping_applied,
    )

    return result, experiments_info, kernel_weights, diagnostics


def filter_exfor_with_energy_bin(
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    bin_lower_mev: float,
    bin_upper_mev: float,
    target_energy_mev: float,
    m_proj_u: float,
    m_targ_u: float,
    dedupe_per_experiment: bool = True,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
) -> Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]:
    """
    Filter EXFOR data using exact energy bin matching.

    Unlike Gaussian kernel weighting, this method:
    - Selects all experiments whose energy falls within [bin_lower, bin_upper]
    - When dedupe_per_experiment=True, selects only the closest energy per experiment
    - Assigns uniform weight = 1.0 to all selected points
    - Does NOT apply per-experiment weight capping

    Parameters
    ----------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Pre-loaded EXFOR data organized by energy
    sorted_energies : List[float]
        Sorted list of available energies in cache (in MeV)
    bin_lower_mev : float
        Lower bin boundary in MeV
    bin_upper_mev : float
        Upper bin boundary in MeV
    target_energy_mev : float
        Target ENDF grid energy in MeV (for diagnostics)
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units
    dedupe_per_experiment : bool
        If True (default), select only the closest energy to target_energy_mev
        for each experiment. This prevents experiments with many energies in
        the bin from dominating the fit.
    exclude_experiments : List[str], optional
        List of experiments to exclude from filtering. Accepts multiple formats:
        - "20743" - excludes all subentries starting with 20743
        - "20743002" - excludes specific dataset
        - "20743/002" - same as above
    min_relative_uncertainty : float, optional
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty.

    Returns
    -------
    Tuple[pd.DataFrame, List[Dict], np.ndarray, KernelDiagnostics]
        - DataFrame with EXFOR data (kernel_weight = 1.0 for all)
        - List of experiment metadata dicts (includes 'selected_from_n_energies')
        - Array of kernel weights (all 1.0)
        - KernelDiagnostics object
    """
    # Empty diagnostics for early returns
    empty_diag = KernelDiagnostics(
        n_eff=0.0, weight_span_95=0.0, weight_span_ratio=0.0,
        n_experiments=0, max_experiment_weight_frac=0.0,
        experiment_weights={}, n_points_dropped=0, capping_applied=False
    )

    if not exfor_cache or not sorted_energies:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Parse exclusion patterns
    exclusion_patterns = _parse_exclusion_list(exclude_experiments)

    all_frames = []
    experiments_info = []

    # Step 1: Collect all candidate data grouped by experiment
    # experiment_candidates: {(entry, subentry): [(energy, df, meta), ...]}
    experiment_candidates: Dict[Tuple[str, str], List[Tuple[float, pd.DataFrame, Dict]]] = defaultdict(list)

    for available_energy in sorted_energies:
        # Exact bin matching - include if within [lower, upper]
        if available_energy < bin_lower_mev or available_energy > bin_upper_mev:
            continue

        entries = exfor_cache.get(available_energy, [])
        for df, meta in entries:
            entry = meta.get('entry', 'unknown')
            subentry = meta.get('subentry', 'unknown')

            # Check if experiment is excluded
            if _is_experiment_excluded(entry, subentry, exclusion_patterns):
                continue

            exp_key = (entry, subentry)
            experiment_candidates[exp_key].append((available_energy, df, meta))

    # Step 2: For each experiment, select closest energy to target (or all if dedupe disabled)
    selected_data: List[Tuple[float, pd.DataFrame, Dict]] = []

    for exp_key, candidates in experiment_candidates.items():
        if dedupe_per_experiment and len(candidates) > 1:
            # Select the energy closest to target
            closest = min(candidates, key=lambda x: abs(x[0] - target_energy_mev))
            selected_data.append(closest)
        else:
            selected_data.extend(candidates)

    # Step 3: Process selected data (transform and build DataFrames)
    for available_energy, df, meta in selected_data:
        # Uniform weight for bin method (no Gaussian decay)
        kernel_weight = 1.0

        # Extract metadata (same as Gaussian kernel method)
        entry = meta.get('entry', 'unknown')
        subentry = meta.get('subentry', 'unknown')
        frame = meta.get('angle_frame', 'CM').upper()
        reaction = meta.get('reaction', '')

        citation = meta.get('citation', {})
        authors = citation.get('authors', [])
        author = authors[0] if authors else 'unknown'
        year = citation.get('year', 'unknown')

        # Extract columns
        angles_deg = df['angle'].to_numpy(dtype=float)
        dsig = df['dsig'].to_numpy(dtype=float)
        error_stat = df['error_stat'].to_numpy(dtype=float)

        n_points = len(angles_deg)

        # Count how many energies this experiment had in the bin
        n_energies_in_bin = len(experiment_candidates[(entry, subentry)])

        # Transform to CM frame if needed (same logic as Gaussian method)
        if frame == 'LAB':
            mu_lab = np.cos(np.deg2rad(angles_deg))
            mu_cm, dsig_cm, error_cm = transform_lab_to_cm(
                mu_lab, dsig, error_stat, m_proj_u, m_targ_u
            )

            angles_cm_deg = np.rad2deg(np.arccos(mu_cm))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_cm_deg,
                'value': dsig_cm,
                'unc': error_cm,
                'mu': mu_cm,
                'frame': 'CM',
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': kernel_weight,
            })
        else:
            mu_cm = np.cos(np.deg2rad(angles_deg))

            transformed_df = pd.DataFrame({
                'theta_deg': angles_deg,
                'value': dsig,
                'unc': error_stat,
                'mu': mu_cm,
                'frame': frame,
                'entry': entry,
                'subentry': subentry,
                'author': author,
                'year': year,
                'reaction': reaction,
                'exfor_energy_mev': available_energy,
                'kernel_weight': kernel_weight,
            })

        all_frames.append(transformed_df)

        # Track experiment info with deduplication info
        exp_info = {
            'entry': entry,
            'subentry': subentry,
            'author': author,
            'year': year,
            'exfor_energy_mev': available_energy,
            'kernel_weight': kernel_weight,
            'n_points': n_points,
            'selected_from_n_energies': n_energies_in_bin,  # NEW: track deduplication
        }
        experiments_info.append(exp_info)

    if not all_frames:
        return pd.DataFrame(), [], np.array([]), empty_diag

    # Concatenate all experiments
    result = pd.concat(all_frames, ignore_index=True)
    kernel_weights = np.ones(len(result), dtype=float)  # All weights = 1.0

    # Apply uncertainty floor if requested
    if min_relative_uncertainty > 0:
        result = apply_uncertainty_floor(result, min_relative_uncertainty, unc_column='unc', value_column='value')

    # NO experiment capping for bin method

    # Compute experiment weight fractions (for logging)
    exp_weight_fracs = {}
    total_points = float(len(result))
    for exp in experiments_info:
        key = f"{exp['entry']}.{exp['subentry']}"
        exp_weight_fracs[key] = exp_weight_fracs.get(key, 0.0) + exp['n_points'] / total_points

    # Compute diagnostics
    # For bin method: N_eff = N (since all weights are equal)
    n_eff = float(len(result))

    # Weight span is the bin width
    weight_span = bin_upper_mev - bin_lower_mev if bin_upper_mev < float('inf') else 0.0

    diagnostics = KernelDiagnostics(
        n_eff=n_eff,
        weight_span_95=weight_span,
        weight_span_ratio=0.0,  # Not applicable for bin method
        n_experiments=len(set(exp_weight_fracs.keys())),
        max_experiment_weight_frac=max(exp_weight_fracs.values()) if exp_weight_fracs else 0.0,
        experiment_weights=exp_weight_fracs,
        n_points_dropped=0,
        capping_applied=False,
    )

    return result, experiments_info, kernel_weights, diagnostics


def apply_min_weight_threshold(
    kernel_weights: np.ndarray,
    min_weight_fraction: float = 1e-3,
) -> Tuple[np.ndarray, int]:
    """
    Zero out kernel weights below threshold.

    Points with g_ij < min_weight_fraction * max(g_ij) are set to zero weight.

    Parameters
    ----------
    kernel_weights : np.ndarray
        Gaussian kernel weights
    min_weight_fraction : float
        Minimum weight as fraction of maximum (default: 1e-3)

    Returns
    -------
    Tuple[np.ndarray, int]
        - filtered_weights: Weights with below-threshold points zeroed
        - n_dropped: Number of points dropped
    """
    if len(kernel_weights) == 0:
        return kernel_weights.copy(), 0

    g_max = np.max(kernel_weights)
    if g_max < 1e-30:
        return kernel_weights.copy(), 0

    threshold = min_weight_fraction * g_max

    filtered = kernel_weights.copy()
    mask = kernel_weights < threshold
    filtered[mask] = 0.0

    return filtered, int(np.sum(mask))


def apply_uncertainty_floor(
    exfor_df: pd.DataFrame,
    min_relative_uncertainty: float = 0.0,
    unc_column: str = "unc",
    value_column: str = "value",
) -> pd.DataFrame:
    """
    Apply minimum relative uncertainty floor to prevent experiments with
    unrealistically small uncertainties from dominating fits.

    For each data point, enforces: unc >= min_relative_uncertainty * |value|

    This is a safety mechanism to handle cases where uncertainties may be
    incorrectly reported or processed in the database.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with uncertainty and value columns
    min_relative_uncertainty : float
        Minimum relative uncertainty as a fraction (default: 0.0 = disabled).
        For example, 0.03 means 3% minimum uncertainty.
    unc_column : str
        Column name for uncertainties (default: 'unc')
    value_column : str
        Column name for cross section values (default: 'value')

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with updated uncertainties (or original if disabled)

    Examples
    --------
    >>> df = apply_uncertainty_floor(exfor_df, min_relative_uncertainty=0.03)
    >>> # Now all points have at least 3% relative uncertainty
    """
    if min_relative_uncertainty <= 0:
        return exfor_df

    df = exfor_df.copy()
    if unc_column not in df.columns or value_column not in df.columns:
        return df

    floor = min_relative_uncertainty * np.abs(df[value_column])
    df[unc_column] = np.maximum(df[unc_column], floor)
    return df


def apply_per_experiment_weight_cap(
    exfor_df: pd.DataFrame,
    kernel_weights: np.ndarray,
    max_experiment_weight_fraction: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float], bool]:
    """
    Cap per-experiment total kernel weight to prevent dense experiments from dominating.

    Parameters
    ----------
    exfor_df : pd.DataFrame
        EXFOR data with 'entry', 'subentry' columns
    kernel_weights : np.ndarray
        Gaussian kernel weights (one per data point)
    max_experiment_weight_fraction : float
        Maximum allowed weight fraction per experiment (default: 0.5)

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float], bool]
        - capped_weights: Adjusted kernel weights
        - experiment_weight_fracs: {exp_key: weight_fraction} BEFORE capping
        - capping_applied: Whether any capping was done
    """
    if max_experiment_weight_fraction >= 1.0:
        # Capping disabled
        return kernel_weights.copy(), {}, False

    if len(kernel_weights) == 0:
        return kernel_weights.copy(), {}, False

    # Build experiment key for each point
    entries = exfor_df['entry'].values
    subentries = exfor_df['subentry'].values
    n_points = len(kernel_weights)
    exp_keys = [f"{entries[i]}.{subentries[i]}" for i in range(n_points)]

    # Compute total weight per experiment
    exp_weights: Dict[str, float] = {}
    for i, key in enumerate(exp_keys):
        exp_weights[key] = exp_weights.get(key, 0.0) + kernel_weights[i]

    total_weight = np.sum(kernel_weights)
    if total_weight < 1e-30:
        return kernel_weights.copy(), {}, False

    # Compute fractions BEFORE capping (for diagnostics)
    exp_weight_fracs = {k: v / total_weight for k, v in exp_weights.items()}

    # Edge case: only one experiment - cannot cap
    if len(exp_weights) == 1:
        return kernel_weights.copy(), exp_weight_fracs, False

    # Apply capping
    capped_weights = kernel_weights.copy()
    capping_applied = False
    cap = max_experiment_weight_fraction

    for exp_key, frac in exp_weight_fracs.items():
        if frac > cap:
            # Scale factor to bring this experiment down to the cap
            scale = cap / frac

            # Apply scale to all points from this experiment
            for i, key in enumerate(exp_keys):
                if key == exp_key:
                    capped_weights[i] *= scale

            capping_applied = True

    return capped_weights, exp_weight_fracs, capping_applied


def load_exfor_with_asymmetric_tolerance(
    exfor_directory: str,
    energy_mev: float,
    tolerance_lower_mev: float,
    tolerance_upper_mev: float,
    m_proj_u: float,
    m_targ_u: float,
) -> Tuple[pd.DataFrame, int]:
    """
    Load EXFOR data with asymmetric tolerance bounds.

    Parameters
    ----------
    exfor_directory : str
        Path to EXFOR data directory
    energy_mev : float
        Target energy in MeV
    tolerance_lower_mev : float
        Lower tolerance in MeV
    tolerance_upper_mev : float
        Upper tolerance in MeV
    m_proj_u : float
        Projectile mass in atomic mass units
    m_targ_u : float
        Target mass in atomic mass units

    Returns
    -------
    Tuple[pd.DataFrame, int]
        DataFrame with EXFOR data and count of unique energies found
    """
    # Use the maximum tolerance for initial search
    max_tolerance = max(tolerance_lower_mev, tolerance_upper_mev)

    # Suppress print statements from load_exfor_for_fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exfor_df = load_exfor_for_fitting(
                exfor_directory=exfor_directory,
                energy_mev=energy_mev,
                tolerance=max_tolerance,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
            )
        finally:
            sys.stdout = old_stdout

    if exfor_df.empty:
        return exfor_df, 0

    # Count unique experiments (entry, subentry pairs)
    if 'entry' in exfor_df.columns and 'subentry' in exfor_df.columns:
        unique_experiments = exfor_df.groupby(['entry', 'subentry']).size()
        n_experiments = len(unique_experiments)
    else:
        n_experiments = 1

    return exfor_df, n_experiments


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_missing_bins(
    results: List[SamplingResult],
    n_samples: int,
) -> List[SamplingResult]:
    """
    Interpolate coefficients for bins where EXFOR data was not available.

    Uses linear interpolation in energy space between neighboring bins
    that have valid data.

    Parameters
    ----------
    results : List[SamplingResult]
        List of sampling results (some may have missing data)
    n_samples : int
        Number of samples expected

    Returns
    -------
    List[SamplingResult]
        Updated results with interpolated coefficients
    """
    logger = _get_logger()

    # Find indices with and without data
    valid_indices = []
    missing_indices = []

    for i, result in enumerate(results):
        if result.sampled_coeffs is not None:
            valid_indices.append(i)
        else:
            missing_indices.append(i)

    if not missing_indices:
        return results

    if len(valid_indices) < 2:
        if logger:
            logger.warning("Not enough valid bins for interpolation - keeping original coefficients for missing bins")
        return results

    # Get energies and coefficient arrays for valid bins
    valid_energies = np.array([results[i].bin_info.energy_mev for i in valid_indices])

    # Determine max coefficient length across all valid bins
    max_n_coeffs = max(results[i].sampled_coeffs.shape[1] for i in valid_indices)

    # Pad coefficient arrays to same size
    valid_coeffs = []
    for i in valid_indices:
        coeffs = results[i].sampled_coeffs
        if coeffs.shape[1] < max_n_coeffs:
            padded = np.zeros((n_samples, max_n_coeffs), dtype=float)
            padded[:, :coeffs.shape[1]] = coeffs
            valid_coeffs.append(padded)
        else:
            valid_coeffs.append(coeffs)
    valid_coeffs = np.array(valid_coeffs)  # Shape: (n_valid, n_samples, n_coeffs)

    # Interpolate for each missing bin
    for miss_idx in missing_indices:
        miss_energy = results[miss_idx].bin_info.energy_mev

        # Check if energy is within interpolation range
        if miss_energy < valid_energies.min() or miss_energy > valid_energies.max():
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV is outside valid range "
                    f"[{valid_energies.min():.4f}, {valid_energies.max():.4f}] - keeping original"
                )
            continue

        # Interpolate each coefficient for each sample
        interp_coeffs = np.zeros((n_samples, max_n_coeffs), dtype=float)

        for sample_idx in range(n_samples):
            for coeff_idx in range(max_n_coeffs):
                # Get values at valid energies for this sample and coefficient
                y_vals = valid_coeffs[:, sample_idx, coeff_idx]
                # Linear interpolation
                interp_coeffs[sample_idx, coeff_idx] = np.interp(
                    miss_energy, valid_energies, y_vals
                )

        results[miss_idx].sampled_coeffs = interp_coeffs
        results[miss_idx].bin_info.interpolated = True

        if logger:
            logger.info(f"E={miss_energy:.4f} MeV: Interpolated coefficients from neighboring bins")

    return results


# =============================================================================
# COVARIANCE COMPUTATION
# =============================================================================

def compute_covariance_from_samples(
    all_samples: Dict[int, Dict[int, np.ndarray]],
    energy_indices: List[int],
    max_order: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute covariance and correlation matrices from MC samples.

    The parameter vector is organized as:
        [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

    Parameters
    ----------
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coeffs}}
    energy_indices : List[int]
        List of energy indices (sorted)
    max_order : int
        Maximum Legendre order to include

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]
        - cov_matrix: Full covariance matrix
        - corr_matrix: Full correlation matrix
        - param_labels: List of (energy_index, order) tuples

    Uncertainty Interpretation
    --------------------------
    The covariance matrix contains variances of ENDF-normalized Legendre
    coefficients a_l = (c_l / c0) / (2l+1). Since these coefficients are
    already normalized by the total cross section (c0), they are dimensionless
    and the variances are inherently RELATIVE (fractional).

    Example: If diag(cov)[k] = 0.0001, then std = 0.01 = 1% fractional uncertainty.

    When written to MF34 with LB=5 format, these values are correctly interpreted
    as relative covariances. The MF34 reader correctly identifies LB=5 data as
    relative (is_relative=True) since LB=0 is the only absolute format.
    """
    n_samples = len(all_samples)
    n_energies = len(energy_indices)
    n_params = n_energies * max_order

    # Build sample matrix
    sample_matrix = np.zeros((n_samples, n_params))

    for s_idx in range(n_samples):
        sample_data = all_samples[s_idx]
        row = []
        for e_idx in energy_indices:
            coeffs = sample_data.get(e_idx, np.zeros(max_order))
            # Pad or truncate to max_order
            padded = np.zeros(max_order)
            padded[:min(len(coeffs), max_order)] = coeffs[:max_order]
            row.extend(padded)
        sample_matrix[s_idx] = row

    # Compute covariance
    cov_matrix = np.cov(sample_matrix, rowvar=False)

    # Compute correlation
    std = np.sqrt(np.diag(cov_matrix))
    std[std == 0] = 1.0  # Avoid division by zero
    corr_matrix = cov_matrix / np.outer(std, std)

    # Generate labels
    param_labels = [(e_idx, l + 1) for e_idx in energy_indices for l in range(max_order)]

    return cov_matrix, corr_matrix, param_labels


def save_all_legendre_coefficients(
    nominal_results: List,  # List[NominalFitResult] - avoid circular import
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
    max_degree: int,
) -> Tuple[str, str]:
    """
    Save all Legendre coefficients (nominal + all MC samples) to file.

    Saves in two formats:
    1. NPZ file (fast loading with numpy)
    2. CSV file (human-readable, easy to import)

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}} for all MC samples
    output_dir : str
        Output directory
    max_degree : int
        Maximum Legendre order

    Returns
    -------
    Tuple[str, str]
        Paths to (npz_file, csv_file)
    """
    output_path = Path(output_dir)

    # Collect all data
    data_rows = []

    # Add nominal coefficients (sample_idx = 0)
    for nr in nominal_results:
        if nr.has_data:
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            # Pad to max_degree if needed
            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': 0,  # 0 = nominal
                'energy_index': nr.energy_index,
                'energy_mev': nr.energy_mev,
            }
            for l in range(max_degree):
                row[f'a_{l+1}'] = padded_coeffs[l]
            data_rows.append(row)

    # Add MC sample coefficients (sample_idx = 1 to N)
    n_samples = len(all_samples)
    for sample_idx in range(n_samples):
        sample_coeffs = all_samples[sample_idx]
        for energy_idx, endf_coeffs in sample_coeffs.items():
            # Find corresponding energy in MeV
            energy_mev = None
            for nr in nominal_results:
                if nr.energy_index == energy_idx:
                    energy_mev = nr.energy_mev
                    break

            # Pad to max_degree if needed
            padded_coeffs = np.zeros(max_degree)
            padded_coeffs[:len(endf_coeffs)] = endf_coeffs

            row = {
                'sample_idx': sample_idx + 1,  # 1-based for MC samples
                'energy_index': energy_idx,
                'energy_mev': energy_mev if energy_mev is not None else 0.0,
            }
            for l in range(max_degree):
                row[f'a_{l+1}'] = padded_coeffs[l]
            data_rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data_rows)

    # Sort by sample_idx, then energy_index
    df = df.sort_values(['sample_idx', 'energy_index']).reset_index(drop=True)

    # Save as CSV
    csv_file = output_path / 'legendre_coefficients_all_samples.csv'
    df.to_csv(csv_file, index=False, float_format='%.6e')

    # Save as NPZ (more compact and faster to load)
    npz_file = output_path / 'legendre_coefficients_all_samples.npz'
    np.savez_compressed(
        npz_file,
        sample_idx=df['sample_idx'].values,
        energy_index=df['energy_index'].values,
        energy_mev=df['energy_mev'].values,
        coefficients=df[[f'a_{l+1}' for l in range(max_degree)]].values,
        max_degree=max_degree,
    )

    return str(npz_file), str(csv_file)


# =============================================================================
# ENDF WRITING FUNCTIONS
# =============================================================================

def write_nominal_endf(
    original_endf_file: str,
    mt_number: int,
    nominal_results: List,  # List[NominalFitResult]
    output_dir: str,
) -> str:
    """
    Write ENDF file with nominal fit coefficients.

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to output file
    """
    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Apply nominal coefficients for energies with EXFOR data
    for nr in nominal_results:
        if nr.has_data and nr.energy_index < len(mt_data._legendre_coeffs):
            # Convert nominal coefficients to ENDF format
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            mt_data._legendre_coeffs[nr.energy_index] = list(endf_coeffs)

    # Create output structure
    output_path = Path(output_dir)
    nominal_dir = output_path / "endf" / "nominal"
    nominal_dir.mkdir(parents=True, exist_ok=True)

    base = Path(original_endf_file).stem
    output_file = nominal_dir / f"{base}_nominal.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    return str(output_file)


def compute_mc_mean_coefficients(
    all_samples: Dict[int, Dict[int, np.ndarray]],
    nominal_results: List,  # List[NominalFitResult]
) -> Dict[int, np.ndarray]:
    """
    Compute MC mean coefficients from all samples.

    Parameters
    ----------
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}}
    nominal_results : List[NominalFitResult]
        Nominal fit results (for energy indices)

    Returns
    -------
    Dict[int, np.ndarray]
        {energy_index: mc_mean_coeffs} in ENDF format (a_1, a_2, ...)
    """
    # Get all energy indices with data
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

    mc_mean_coeffs = {}
    n_samples = len(all_samples)

    for e_idx in energy_indices:
        # Collect coefficients from all samples for this energy
        sample_coeffs_list = []
        for s_idx in range(n_samples):
            if e_idx in all_samples[s_idx]:
                sample_coeffs_list.append(all_samples[s_idx][e_idx])

        if sample_coeffs_list:
            # Stack and compute mean
            stacked = np.vstack(sample_coeffs_list)
            mc_mean_coeffs[e_idx] = np.mean(stacked, axis=0)

    return mc_mean_coeffs


def write_evaluation_endf(
    original_endf_file: str,
    mt_number: int,
    nominal_results: List,  # List[NominalFitResult]
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
) -> str:
    """
    Write ENDF file with MC mean coefficients (evaluation file).

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    nominal_results : List[NominalFitResult]
        Nominal fit results from Phase 1
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: endf_coeffs}} for all MC samples
    output_dir : str
        Output directory

    Returns
    -------
    str
        Path to output file
    """
    # Compute MC mean coefficients
    mc_mean_coeffs = compute_mc_mean_coefficients(all_samples, nominal_results)

    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Apply MC mean coefficients for energies with data
    for e_idx, mean_coeffs in mc_mean_coeffs.items():
        if e_idx < len(mt_data._legendre_coeffs):
            mt_data._legendre_coeffs[e_idx] = list(mean_coeffs)

    # Create output structure
    output_path = Path(output_dir)
    eval_dir = output_path / "endf" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    base = Path(original_endf_file).stem
    output_file = eval_dir / f"{base}_evaluation.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    return str(output_file)


def write_endf_sample(
    sample_index: int,
    original_endf_file: str,
    mt_number: int,
    energy_indices: List[int],
    sampled_coeffs_by_energy: Dict[int, np.ndarray],
    output_dir: str,
    cached_original_coeffs: Optional[List[List[float]]] = None,
) -> str:
    """
    Write a single ENDF sample file.

    Parameters
    ----------
    sample_index : int
        Sample index (0-based)
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    energy_indices : List[int]
        Indices of energies to modify (unused, kept for compatibility)
    sampled_coeffs_by_energy : Dict[int, np.ndarray]
        Coefficients for each energy index (for this sample)
    output_dir : str
        Output directory
    cached_original_coeffs : Optional[List[List[float]]]
        If provided, use these as the original coefficients

    Returns
    -------
    str
        Path to output file
    """
    # Parse original ENDF
    endf = read_endf(original_endf_file)
    mf4 = endf.get_file(4)

    if mf4 is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data = mf4.sections.get(mt_number)
    if mt_data is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    # Check type
    if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Modify coefficients at each energy
    for energy_idx, new_coeffs in sampled_coeffs_by_energy.items():
        if energy_idx < len(mt_data._legendre_coeffs):
            mt_data._legendre_coeffs[energy_idx] = list(new_coeffs)

    # Write output file
    sample_str = f"{sample_index + 1:04d}"
    base = Path(original_endf_file).stem

    # Create output structure
    sample_dir = Path(output_dir) / "endf" / sample_str
    sample_dir.mkdir(parents=True, exist_ok=True)

    output_file = sample_dir / f"{base}_{sample_str}.endf"

    # Use ENDFWriter
    writer = ENDFWriter(original_endf_file)
    success = writer.replace_mf_section(mf4, str(output_file))

    if not success:
        raise RuntimeError(f"Failed to write {output_file}")

    return str(output_file)


def _write_sample_wrapper(args):
    """Wrapper for parallel writing of ENDF samples."""
    return write_endf_sample(*args)


def write_endf_samples_batch(
    original_endf_file: str,
    mt_number: int,
    all_samples: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
) -> List[str]:
    """
    Write multiple ENDF samples efficiently (sequential mode).

    Reads the ENDF file ONCE and reuses the parsed structure for all samples.

    Parameters
    ----------
    original_endf_file : str
        Path to original ENDF file
    mt_number : int
        MT reaction number
    all_samples : Dict[int, Dict[int, np.ndarray]]
        {sample_idx: {energy_index: coefficients}}
    output_dir : str
        Output directory

    Returns
    -------
    List[str]
        Paths to output files
    """
    output_path = Path(output_dir)
    base = Path(original_endf_file).stem

    # Parse ENDF file ONCE
    endf_template = read_endf(original_endf_file)
    mf4_template = endf_template.get_file(4)

    if mf4_template is None:
        raise ValueError(f"MF4 not found in {original_endf_file}")

    mt_data_template = mf4_template.sections.get(mt_number)
    if mt_data_template is None:
        raise ValueError(f"MT{mt_number} not found in MF4")

    if not isinstance(mt_data_template, (MF4MTLegendre, MF4MTMixed)):
        raise ValueError(f"MT{mt_number} is not Legendre or Mixed type")

    # Store original coefficients for restoration
    original_coeffs = [list(c) for c in mt_data_template._legendre_coeffs]

    # Create writer once
    writer = ENDFWriter(original_endf_file)

    output_files = []
    n_samples = len(all_samples)

    for sample_idx in sorted(all_samples.keys()):
        # Restore original coefficients
        mt_data_template._legendre_coeffs = [list(c) for c in original_coeffs]

        # Apply sampled coefficients
        sampled_coeffs = all_samples[sample_idx]
        for energy_idx, new_coeffs in sampled_coeffs.items():
            if energy_idx < len(mt_data_template._legendre_coeffs):
                mt_data_template._legendre_coeffs[energy_idx] = list(new_coeffs)

        # Write output file
        sample_str = f"{sample_idx + 1:04d}"
        sample_dir = output_path / "endf" / sample_str
        sample_dir.mkdir(parents=True, exist_ok=True)
        output_file = sample_dir / f"{base}_{sample_str}.endf"

        success = writer.replace_mf_section(mf4_template, str(output_file))
        if not success:
            raise RuntimeError(f"Failed to write {output_file}")

        output_files.append(str(output_file))

        if (sample_idx + 1) % 50 == 0 or sample_idx == 0 or sample_idx == n_samples - 1:
            print(f"[INFO] Writing sample {sample_idx + 1}/{n_samples}")

    return output_files


# =============================================================================
# MF34 COVARIANCE FUNCTIONS
# =============================================================================
# These functions are now provided by kika.endf.writers module.
# Import and re-export for backward compatibility with existing scripts.

from kika.endf.writers import (
    create_mf34_from_covariance,
    write_mf34_to_file as write_mf34_to_endf,  # Alias for backward compatibility
)
