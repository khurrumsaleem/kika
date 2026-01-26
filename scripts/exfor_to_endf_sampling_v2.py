"""
EXFOR-to-ENDF Angular Distribution Sampling Script (v2 - Using kika.exfor module).

This script generates N samples of ENDF files by fitting Legendre coefficients
to EXFOR experimental angular distribution data at each energy point in the
original ENDF's energy grid, then replacing the original coefficients with
sampled values.

This is the migrated version that uses:
- kika.exfor module for EXFOR data loading (read_all_exfor)
- kika.exfor.transforms for frame conversions
- kika.endf.writers for MF34 creation

The workflow:
1. Read the reference ENDF file and extract the MF4 Legendre energy grid
2. For each energy bin in the specified range, compute adaptive tolerance
   based on neighboring energy points
3. Load EXFOR data within tolerance and fit Legendre polynomials
4. Generate N samples of coefficients using deterministic seeds for energy correlation
5. Create N output ENDF files with the sampled coefficients
6. Handle missing data by interpolating from neighboring bins

Author: Generated for kika project
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from multiprocessing import Pool
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Add kika to path if needed (from scripts/ directory, kika is parent)
_kika_path = Path(__file__).parent.parent
if str(_kika_path) not in sys.path:
    sys.path.insert(0, str(_kika_path))

# Import kika modules - ENDF reading
from kika.endf.read_endf import read_endf
from kika.endf.classes.mf4.polynomial import MF4MTLegendre
from kika.endf.classes.mf4.mixed import MF4MTMixed

# Import kika modules - NEW EXFOR module (replaces AD_utils.load_all_exfor_data)
from kika.exfor import read_all_exfor

# Import kika modules - MF34 from library (replaces local implementation)
from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file

# Import local utility module (uses relative import from scripts package)
from scripts.exfor_utils import (
    # Logging
    DualLogger,
    _get_logger,
    _set_logger,
    _format_condensed_experiments,
    # Data classes
    EnergyBinInfo,
    SamplingResult,
    KernelDiagnostics,
    # Energy binning
    compute_energy_bins_with_tof_resolution,
    # EXFOR data conversion (new API -> legacy format)
    build_exfor_cache_from_objects,
    # EXFOR filtering (per-energy methods)
    filter_exfor_with_kernel_weights,
    filter_exfor_with_energy_bin,
    # Covariance
    compute_covariance_from_samples,
    save_all_legendre_coefficients,
    # ENDF writing
    write_nominal_endf,
    write_evaluation_endf,
    write_endf_samples_batch,
    write_endf_sample,
    _write_sample_wrapper,
)

# Import resample_AD functions (relative import from scripts package)
from scripts.resample_AD import (
    sample_legendre_coefficients,
    endf_normalize_legendre_coeffs,
    compute_angular_band_discrepancy,
    smooth_tau_in_energy,
    compute_n_eff,
    fit_legendre_global_convolution,
    GlobalFitDiagnostics,
)

import time


# =============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE BEFORE RUNNING
# =============================================================================

# -----------------------------------------------------------------------------
# 1. INPUT/OUTPUT PATHS
# -----------------------------------------------------------------------------
# Reference ENDF file (source of energy grid and original Legendre coefficients)
ENDF_FILE = "/soft_snc/lib/endf/jeff40/neutrons/26-Fe-56g.txt"

# EXFOR JSON directory (for source="json" or "auto")
EXFOR_DIRECTORY = "/share_snc/snc/JuanMonleon/EXFOR/data_v1/"

# X4Pro SQLite database path (for source="database" or "auto")
# Set to None to use KIKA_X4PRO_DB_PATH env variable or builtin default
EXFOR_DB_PATH = '/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db'

# Output directory (all generated files go here)
OUTPUT_DIR = "/SCRATCH/users/monleon-de-la-jan/MCNPy_LIB/EXFOR_FIT_JEFF_V2_ENERGYBIN/"

# -----------------------------------------------------------------------------
# 2. DATA SOURCE CONFIGURATION
# -----------------------------------------------------------------------------
# Data source: "json", "database", "auto" (database + JSON fallback), or "both"
EXFOR_SOURCE = "database"

# Filter options for database queries
# Use list of ZAIDs to include both Fe-56 and natural iron (Fe-0) experiments
# Natural iron is ~92% Fe-56 and has much more experimental coverage in 1-3 MeV range
TARGET_ZAIDS = [26056, 26000]                    # Target ZAIDs: Fe-56 + natural iron
TARGET_PROJECTILE = "N"                          # Projectile (N for neutrons)

# Supplementary JSON files (for experiments not in database)
# These files will be loaded in addition to the main data source
SUPPLEMENTARY_JSON_FILES = [
    '/share_snc/snc/JuanMonleon/EXFOR/data_v1/27673002.json',
    # "C:/Users/Usuario/BaradDur/EXFOR/data_v1/data_v1/27673002.json",  # Gkatis (2025)
]

# -----------------------------------------------------------------------------
# 3. OUTPUT GENERATION OPTIONS
# -----------------------------------------------------------------------------
GENERATE_NOMINAL_ENDF = True                     # Best-fit coefficients ENDF
GENERATE_MC_MEAN_ENDF = True                     # MC mean coefficients ENDF
GENERATE_SAMPLES_ENDF = False                    # Individual MC sample ENDFs
GENERATE_COVARIANCE = True                      # Covariance matrix (.npy)
GENERATE_MF34 = True                            # MF34 covariance section in ENDF
N_SAMPLES = 10                                   # Number of MC samples

# -----------------------------------------------------------------------------
# 4. GENERAL PARAMETERS (Apply to ALL methods)
# -----------------------------------------------------------------------------
# Energy range to process (in MeV)
ENERGY_MIN_MEV = 1.0
ENERGY_MAX_MEV = 3.0

# MT reaction number (2 = elastic scattering)
MT_NUMBER = 2

# Target isotope masses (for LAB->CM frame conversion)
M_PROJ_U = 1.008665                              # Projectile mass in u (neutron)
M_TARG_U = 55.93494                              # Target mass in u (Fe-56)

# Legendre fitting parameters
MAX_LEGENDRE_DEGREE = 8                          # Maximum Legendre order (capped at 8)
SELECT_DEGREE = "aicc"                           # "aicc", "bic", or None (use max)
RIDGE_LAMBDA = 1e-6                              # Ridge regularization parameter

# Processing options
N_PROCS = 5                                      # Parallel processes (1 = sequential)
BASE_SEED = 42                                   # Random seed for reproducibility

# -----------------------------------------------------------------------------
# 5. EXPERIMENT SELECTION METHOD
# -----------------------------------------------------------------------------
# Available methods:
#
# "global_convolution" (RECOMMENDED)
#     Fits ALL energy points simultaneously using Tikhonov regularization.
#     Properly accounts for energy resolution smearing across energy bins.
#     Each EXFOR measurement contributes to multiple ENDF energies according
#     to its resolution-weighted probability. Enforces smooth energy dependence.
#     --> Uses parameters from section 6a (TOF Resolution) and 6b (Global Convolution)
#
# "kernel_weights"
#     Fits each ENDF energy point independently using Gaussian kernel weighting.
#     EXFOR data are weighted by g_ij = exp(-0.5 * ((E_i - E_j)/σE)²)
#     --> Uses parameters from sections 6a, 6c, 6d, 6e, 6f
#
# "energy_bin"
#     Simple energy binning without resolution-based weighting.
#     Uses hard bin boundaries (no Gaussian weighting).
#     Fastest but ignores energy resolution effects.
#     --> Uses parameters from sections 6d, 6e, 6f
#
EXPERIMENT_SELECTION_METHOD = "energy_bin"

# --- Experiment Exclusion and Uncertainty Floor ---
# Experiments to exclude from fitting (e.g., experiments with known issues)
# Accepts formats: "20743" (all subentries), "20743002", or "20743/002"
EXCLUDE_EXPERIMENTS = ["20743002"]  # e.g., ["20743002"] to exclude Cierjacks (1978)

# Minimum relative uncertainty floor (prevents unrealistically small errors from dominating)
# Set to 0.0 to disable. e.g., 0.03 for 3% minimum uncertainty
MIN_RELATIVE_UNCERTAINTY = 0.03

# -----------------------------------------------------------------------------
# 6. METHOD-SPECIFIC PARAMETERS
# -----------------------------------------------------------------------------

# --- 6a. TOF Energy Resolution (kernel_weights, global_convolution) ---
DELTA_T_NS = 5.0                                 # Time resolution in nanoseconds
FLIGHT_PATH_M = 27.037                           # Flight path in meters
N_SIGMA_CUTOFF = 3.0                             # Gaussian kernel cutoff (±n_sigma * σE)

# --- 6b. Global Convolution (EXPERIMENT_SELECTION_METHOD = "global_convolution") ---
GLOBAL_CONV_LAMBDA = 0.001                       # Tikhonov regularization strength

# --- 6c. Kernel Weight Control (EXPERIMENT_SELECTION_METHOD = "kernel_weights") ---
MIN_KERNEL_WEIGHT_FRACTION = 1e-3                # Minimum weight threshold
MAX_EXPERIMENT_WEIGHT_FRACTION = 0.5             # Weight cap per experiment
N_EFF_WARNING_THRESHOLD = 5.0                    # Warning if N_eff < threshold
WEIGHT_SPAN_WARNING_RATIO = 3.0                  # Warning if span > ratio * σE

# --- 6d. Angular-Band Discrepancy (kernel_weights, energy_bin) ---
USE_BAND_DISCREPANCY = True                      # Use band-based uncertainty (vs global Birge)
MIN_POINTS_PER_BAND = 3                          # Minimum points to estimate τ_b per band
MAX_TAU_FRACTION = 0.25                          # Cap τ_b at 25% of cross section
TAU_SMOOTHING_WINDOW = 3                         # Moving median window for τ_b(E) smoothing

# --- 6e. Per-Experiment Normalization (kernel_weights, energy_bin) ---
NORMALIZATION_SIGMA = 0.05                       # Per-experiment normalization uncertainty (5%)

# --- 6f. Model Averaging (kernel_weights, energy_bin) ---
USE_MODEL_AVERAGING = True                       # Enable model averaging over Legendre orders
MIN_DEGREE_FOR_AVERAGING = 1                     # Minimum degree to consider (1 = include all)

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


# Global logger reference (set by _set_logger from exfor_utils)
_logger = None


# =============================================================================
# WORKFLOW DATACLASSES
# =============================================================================

@dataclass
class NominalFitResult:
    """Result from nominal fitting at one energy."""
    energy_mev: float
    energy_index: int
    exfor_df: pd.DataFrame
    experiments_info: List[Dict]
    kernel_weights: np.ndarray
    frozen_degree: int
    nominal_coeffs: np.ndarray
    sigma_eff: np.ndarray
    tau_info: Dict[str, float]
    chi2_red: float
    has_data: bool = True
    interpolated: bool = False  # Whether coefficients were interpolated from neighbors
    kernel_diagnostics: Optional[KernelDiagnostics] = None
    degree_weights: Optional[Dict[int, float]] = None
    all_degrees_info: Optional[Dict[int, Dict]] = None


# Import the rest of the workflow functions from the original script
# These are the same as the original - just need to update the MF34 calls


def interpolate_missing_nominal_fits(
    nominal_results: List[NominalFitResult],
    logger=None,
) -> List[NominalFitResult]:
    """
    Interpolate coefficients for bins where EXFOR data was not available.

    IMPORTANT: This ensures we NEVER use original ENDF coefficients as fallback,
    which would compromise the independence of the evaluation.

    Uses linear interpolation in energy space between neighboring bins
    that have valid EXFOR data.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        List of nominal fit results (some may have has_data=False)
    logger : optional
        Logger instance

    Returns
    -------
    List[NominalFitResult]
        Updated results with interpolated coefficients for missing bins
    """
    # Find indices with and without data
    valid_indices = []
    missing_indices = []

    for i, result in enumerate(nominal_results):
        if result.has_data:
            valid_indices.append(i)
        else:
            missing_indices.append(i)

    if not missing_indices:
        return nominal_results

    if len(valid_indices) < 2:
        if logger:
            logger.error(
                "CRITICAL: Not enough bins with EXFOR data for interpolation. "
                "Cannot proceed without at least 2 bins with data."
            )
            logger.error(
                "  -> Energy bins without EXFOR data will have ISOTROPIC (a0=1) coefficients."
            )
            logger.error(
                "  -> These bins should be excluded from final evaluation or data coverage improved."
            )
        # Set isotropic for missing bins rather than using original ENDF
        for miss_idx in missing_indices:
            nominal_results[miss_idx].nominal_coeffs = np.array([1.0])
            nominal_results[miss_idx].frozen_degree = 0
        return nominal_results

    # Get energies and coefficient arrays for valid bins
    valid_energies = np.array([nominal_results[i].energy_mev for i in valid_indices])

    # Determine max coefficient length across all valid bins
    max_n_coeffs = max(len(nominal_results[i].nominal_coeffs) for i in valid_indices)

    # Pad coefficient arrays to same size
    valid_coeffs = []
    for i in valid_indices:
        coeffs = nominal_results[i].nominal_coeffs
        if len(coeffs) < max_n_coeffs:
            padded = np.zeros(max_n_coeffs, dtype=float)
            padded[:len(coeffs)] = coeffs
            valid_coeffs.append(padded)
        else:
            valid_coeffs.append(coeffs)
    valid_coeffs = np.array(valid_coeffs)  # Shape: (n_valid, n_coeffs)

    # Also get frozen degrees for interpolation (round to nearest integer)
    valid_degrees = np.array([nominal_results[i].frozen_degree for i in valid_indices])

    # Interpolate for each missing bin
    n_interpolated = 0
    n_extrapolated = 0

    for miss_idx in missing_indices:
        miss_energy = nominal_results[miss_idx].energy_mev

        # Check if energy is within interpolation range
        if miss_energy < valid_energies.min():
            # Extrapolate below - use lowest valid bin's values
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Below EXFOR data range [{valid_energies.min():.4f} MeV] "
                    f"- extrapolating from lowest valid bin"
                )
            # Use nearest (lowest energy with data)
            nearest_idx = valid_indices[0]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
            n_extrapolated += 1
            continue

        if miss_energy > valid_energies.max():
            # Extrapolate above - use highest valid bin's values
            if logger:
                logger.warning(
                    f"E={miss_energy:.4f} MeV: Above EXFOR data range [{valid_energies.max():.4f} MeV] "
                    f"- extrapolating from highest valid bin"
                )
            # Use nearest (highest energy with data)
            nearest_idx = valid_indices[-1]
            nominal_results[miss_idx].nominal_coeffs = nominal_results[nearest_idx].nominal_coeffs.copy()
            nominal_results[miss_idx].frozen_degree = nominal_results[nearest_idx].frozen_degree
            nominal_results[miss_idx].interpolated = True
            nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
            n_extrapolated += 1
            continue

        # Interpolate each coefficient
        interp_coeffs = np.zeros(max_n_coeffs, dtype=float)
        for coeff_idx in range(max_n_coeffs):
            y_vals = valid_coeffs[:, coeff_idx]
            interp_coeffs[coeff_idx] = np.interp(miss_energy, valid_energies, y_vals)

        # Interpolate degree (round to nearest integer)
        interp_degree = int(round(np.interp(miss_energy, valid_energies, valid_degrees.astype(float))))

        nominal_results[miss_idx].nominal_coeffs = interp_coeffs
        nominal_results[miss_idx].frozen_degree = interp_degree
        nominal_results[miss_idx].interpolated = True
        nominal_results[miss_idx].has_data = True  # Mark as having data (interpolated)
        n_interpolated += 1

        if logger:
            logger.info(
                f"E={miss_energy:.4f} MeV: INTERPOLATED from neighboring bins (L={interp_degree})"
            )

    if logger:
        logger.info("")
        logger.info("[INTERPOLATION SUMMARY]")
        logger.info(f"  Bins with EXFOR data: {len(valid_indices)}")
        logger.info(f"  Bins interpolated: {n_interpolated}")
        logger.info(f"  Bins extrapolated: {n_extrapolated}")
        if n_extrapolated > 0:
            logger.warning(
                f"  WARNING: {n_extrapolated} bins were extrapolated outside EXFOR data range. "
                f"Consider expanding energy range or improving data coverage."
            )

    return nominal_results


def log_experiments_summary(
    nominal_results: List[NominalFitResult],
    logger=None,
) -> None:
    """
    Log a summary of all EXFOR experiments used across all energy bins.

    This provides a quick overview of data sources used in the evaluation.

    Parameters
    ----------
    nominal_results : List[NominalFitResult]
        List of nominal fit results
    logger : optional
        Logger instance
    """
    from collections import defaultdict

    if not logger:
        return

    # Aggregate experiments across all energy bins
    experiment_totals = defaultdict(lambda: {
        'author': '',
        'year': '',
        'n_energies': 0,
        'total_points': 0,
        'energy_min': float('inf'),
        'energy_max': float('-inf'),
    })

    for nr in nominal_results:
        if not nr.has_data or nr.interpolated:
            continue  # Skip interpolated bins - they don't contribute new data

        for exp in nr.experiments_info:
            key = (exp['entry'], exp['subentry'])
            experiment_totals[key]['author'] = exp.get('author', 'Unknown')
            experiment_totals[key]['year'] = exp.get('year', '????')
            experiment_totals[key]['n_energies'] += 1
            experiment_totals[key]['total_points'] += exp.get('n_points', 0)

            exp_energy = exp.get('exfor_energy_mev', nr.energy_mev)
            if exp_energy < experiment_totals[key]['energy_min']:
                experiment_totals[key]['energy_min'] = exp_energy
            if exp_energy > experiment_totals[key]['energy_max']:
                experiment_totals[key]['energy_max'] = exp_energy

    if not experiment_totals:
        logger.info("[EXPERIMENTS USED - SUMMARY]")
        logger.info("  No experiments found (all bins interpolated or no data)")
        return

    # Sort by total points (descending)
    sorted_experiments = sorted(
        experiment_totals.items(),
        key=lambda x: x[1]['total_points'],
        reverse=True
    )

    # Calculate totals
    total_experiments = len(sorted_experiments)
    total_points = sum(exp['total_points'] for _, exp in sorted_experiments)
    total_energy_bins = sum(1 for nr in nominal_results if nr.has_data and not nr.interpolated)

    logger.info("")
    logger.info("=" * 80)
    logger.info("[EXPERIMENTS USED - SUMMARY]")
    logger.info("=" * 80)
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Total data points: {total_points}")
    logger.info(f"  Energy bins with EXFOR data: {total_energy_bins}")
    logger.info("")
    logger.info("  Experiment details (sorted by total points):")
    logger.info("  " + "-" * 76)
    logger.info(f"  {'Entry.Sub':<12} {'Author':<20} {'Year':<6} {'Energies':<10} {'Points':<8} {'E range (MeV)'}")
    logger.info("  " + "-" * 76)

    for (entry, subentry), data in sorted_experiments:
        exp_id = f"{entry}.{subentry}"
        author = data['author'][:18] if len(data['author']) > 18 else data['author']
        year = str(data['year'])
        n_energies = data['n_energies']
        total_pts = data['total_points']

        if data['energy_min'] == data['energy_max']:
            e_range = f"{data['energy_min']:.4f}"
        else:
            e_range = f"{data['energy_min']:.4f}-{data['energy_max']:.4f}"

        logger.info(f"  {exp_id:<12} {author:<20} {year:<6} {n_energies:<10} {total_pts:<8} {e_range}")

    logger.info("  " + "-" * 76)
    logger.info(f"  {'TOTAL':<12} {'':<20} {'':<6} {'':<10} {total_points:<8}")
    logger.info("=" * 80)
    logger.info("")


def perform_nominal_fits(
    energy_bins: List[EnergyBinInfo],
    exfor_cache: Dict[float, List[Tuple[pd.DataFrame, Dict]]],
    sorted_energies: List[float],
    n_sigma: float,
    max_degree: int,
    select_degree: Optional[str],
    ridge_lambda: float,
    m_proj_u: float,
    m_targ_u: float,
    use_band_discrepancy: bool,
    min_points_per_band: int,
    max_tau_fraction: float,
    tau_smoothing_window: int,
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    n_eff_warning_threshold: float = 5.0,
    weight_span_warning_ratio: float = 3.0,
    experiment_selection_method: str = "global_convolution",
    min_degree_for_averaging: int = 3,
    delta_t_ns: float = 10.0,
    flight_path_m: float = 27.037,
    tikhonov_lambda: float = 0.001,
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
    logger = None,
) -> List[NominalFitResult]:
    """Phase 1: Perform nominal fits to determine frozen orders and band discrepancies."""
    from numpy.polynomial.legendre import legvander, legval

    logger = _get_logger()
    results = []

    # GLOBAL CONVOLUTION METHOD
    if experiment_selection_method == "global_convolution":
        if logger:
            logger.info("")
            logger.info("[GLOBAL CONVOLUTION FIT]")
            logger.info(f"  Method: global_convolution (all energies fitted simultaneously)")
            logger.info(f"  Tikhonov λ: {tikhonov_lambda}")
            logger.info(f"  Max Legendre degree: {max_degree}")
            logger.info(f"  Energy kernel cutoff: ±{n_sigma}σ")
            logger.info(f"  Min weight fraction: {min_kernel_weight_fraction}")
            logger.info("")

        coeffs_by_energy, global_diag = fit_legendre_global_convolution(
            exfor_cache=exfor_cache,
            sorted_energies=sorted_energies,
            energy_bins=energy_bins,
            max_degree=max_degree,
            n_sigma=n_sigma,
            tikhonov_lambda=tikhonov_lambda,
            min_kernel_weight_fraction=min_kernel_weight_fraction,
            m_proj_u=m_proj_u,
            m_targ_u=m_targ_u,
            delta_t_ns=delta_t_ns,
            flight_path_m=flight_path_m,
            logger=logger,
        )

        for bin_info in energy_bins:
            if bin_info.index in coeffs_by_energy:
                coeffs = coeffs_by_energy[bin_info.index]
                chi2_red = global_diag.chi2_per_energy.get(bin_info.index, 0.0)
                n_eff = global_diag.n_eff_per_energy.get(bin_info.index, 0.0)

                kernel_diag = KernelDiagnostics(
                    n_eff=n_eff,
                    weight_span_95=0.0,
                    weight_span_ratio=0.0,
                    n_experiments=0,
                    max_experiment_weight_frac=0.0,
                    experiment_weights={},
                    n_points_dropped=0,
                    capping_applied=False,
                )

                results.append(NominalFitResult(
                    energy_mev=bin_info.energy_mev,
                    energy_index=bin_info.index,
                    exfor_df=pd.DataFrame(),
                    experiments_info=[],
                    kernel_weights=np.array([]),
                    frozen_degree=max_degree,
                    nominal_coeffs=coeffs,
                    sigma_eff=np.array([]),
                    tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                    chi2_red=chi2_red,
                    has_data=True,
                    kernel_diagnostics=kernel_diag,
                    degree_weights=None,
                    all_degrees_info=None,
                ))
            else:
                results.append(NominalFitResult(
                    energy_mev=bin_info.energy_mev,
                    energy_index=bin_info.index,
                    exfor_df=pd.DataFrame(),
                    experiments_info=[],
                    kernel_weights=np.array([]),
                    frozen_degree=0,
                    nominal_coeffs=np.array([1.0]),
                    sigma_eff=np.array([]),
                    tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                    chi2_red=0.0,
                    has_data=False,
                    kernel_diagnostics=None,
                    degree_weights=None,
                    all_degrees_info=None,
                ))

        if logger:
            n_with_data = len(global_diag.energies_with_data)
            logger.info("[GLOBAL FIT SUMMARY]")
            logger.info(f"  Energies with data: {n_with_data}/{len(energy_bins)}")
            logger.info(f"  Total χ² = {global_diag.chi2:.2f}")
            if hasattr(global_diag, 'n_eff_total') and global_diag.n_eff_total:
                logger.info(f"  Total N_eff = {global_diag.n_eff_total:.1f}")
            if hasattr(global_diag, 'n_points_total') and global_diag.n_points_total:
                logger.info(f"  Total data points = {global_diag.n_points_total}")
            logger.info("")
            # Log per-energy results in condensed form
            logger.info("  Per-energy results:")
            for bin_info in energy_bins:
                if bin_info.index in coeffs_by_energy:
                    chi2_e = global_diag.chi2_per_energy.get(bin_info.index, 0.0)
                    n_eff_e = global_diag.n_eff_per_energy.get(bin_info.index, 0.0)
                    logger.info(
                        f"    E={bin_info.energy_mev:.4f} MeV: χ²/dof={chi2_e:.2f}, N_eff={n_eff_e:.1f}"
                    )
                else:
                    logger.info(f"    E={bin_info.energy_mev:.4f} MeV: No data")
            logger.info("")

        return results

    # PER-ENERGY METHODS
    for bin_info in energy_bins:
        if experiment_selection_method == "energy_bin":
            exfor_df, experiments_info, kernel_weights, diagnostics = filter_exfor_with_energy_bin(
                exfor_cache=exfor_cache,
                sorted_energies=sorted_energies,
                bin_lower_mev=bin_info.bin_lower_mev,
                bin_upper_mev=bin_info.bin_upper_mev,
                target_energy_mev=bin_info.energy_mev,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
                dedupe_per_experiment=True,
                exclude_experiments=exclude_experiments,
                min_relative_uncertainty=min_relative_uncertainty,
            )
        else:
            exfor_df, experiments_info, kernel_weights, diagnostics = filter_exfor_with_kernel_weights(
                exfor_cache=exfor_cache,
                sorted_energies=sorted_energies,
                energy_mev=bin_info.energy_mev,
                sigma_E_mev=bin_info.sigma_E_mev,
                n_sigma=n_sigma,
                m_proj_u=m_proj_u,
                m_targ_u=m_targ_u,
                bin_lower_mev=bin_info.bin_lower_mev,
                bin_upper_mev=bin_info.bin_upper_mev,
                min_kernel_weight_fraction=min_kernel_weight_fraction,
                max_experiment_weight_fraction=max_experiment_weight_fraction,
                default_delta_t_ns=delta_t_ns,
                default_flight_path_m=flight_path_m,
                use_overlap_weights=False,
                normalize_by_n_points=False,
                dedupe_per_experiment=True,
                exclude_experiments=exclude_experiments,
                min_relative_uncertainty=min_relative_uncertainty,
                logger=logger,
            )

        if exfor_df.empty or len(exfor_df) < 3:
            results.append(NominalFitResult(
                energy_mev=bin_info.energy_mev,
                energy_index=bin_info.index,
                exfor_df=pd.DataFrame(),
                experiments_info=[],
                kernel_weights=np.array([]),
                frozen_degree=0,
                nominal_coeffs=np.array([1.0]),
                sigma_eff=np.array([]),
                tau_info={'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
                chi2_red=0.0,
                has_data=False,
                kernel_diagnostics=None,
            ))
            if logger:
                logger.warning(f"E={bin_info.energy_mev:.4f} MeV: No EXFOR data (σE={bin_info.sigma_E_mev:.4f} MeV)")
            continue

        bin_info.has_exfor_data = True
        bin_info.exfor_n_points = len(exfor_df)
        bin_info.exfor_n_experiments = len(experiments_info)
        bin_info.experiments_used = experiments_info

        mu = exfor_df['mu'].to_numpy()
        y = exfor_df['value'].to_numpy()
        sigma = exfor_df['unc'].to_numpy()

        coef_df, fit_info = sample_legendre_coefficients(
            exfor_df,
            value_col="value",
            unc_col="unc",
            degree=None,
            max_degree=max_degree,
            select_degree=select_degree,
            ridge_lambda=ridge_lambda,
            external_weights=kernel_weights,
            n_samples=1,
            use_band_discrepancy=use_band_discrepancy,
            min_points_per_band=min_points_per_band,
            max_tau_fraction=max_tau_fraction,
        )

        frozen_degree = fit_info['degree']
        nominal_coeffs = coef_df.iloc[0].to_numpy()
        chi2_red = fit_info['chi2_red']
        tau_info = fit_info.get('tau_info', {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0})

        all_degrees_info = fit_info.get('all_degrees_info', None)
        degree_weights = None

        if all_degrees_info and len(all_degrees_info) > 1:
            aicc_values = {d: info['aicc'] for d, info in all_degrees_info.items()}
            min_aicc = min(aicc_values.values())
            raw_weights = {d: np.exp(-0.5 * (aicc - min_aicc)) for d, aicc in aicc_values.items()}
            total = sum(raw_weights.values())
            degree_weights = {d: w / total for d, w in raw_weights.items()}
            degree_weights = {d: w for d, w in degree_weights.items() if w > 0.01 and d >= min_degree_for_averaging}
            if degree_weights:
                total = sum(degree_weights.values())
                degree_weights = {d: w / total for d, w in degree_weights.items()}
            else:
                degree_weights = {frozen_degree: 1.0}

        y_fit = legval(mu, nominal_coeffs)

        if use_band_discrepancy and tau_info:
            sigma_eff, _ = compute_angular_band_discrepancy(
                mu=mu, y=y, sigma=sigma, y_fit=y_fit,
                min_points_per_band=min_points_per_band,
                max_tau_fraction=max_tau_fraction,
            )
            tau_F = tau_info.get('tau_F', 0.0)
            tau_M = tau_info.get('tau_M', 0.0)
            tau_B = tau_info.get('tau_B', 0.0)
        else:
            scale = max(1.0, np.sqrt(chi2_red))
            sigma_eff = sigma * scale
            tau_F = tau_M = tau_B = 0.0
            tau_info = {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0}

        bin_info.fitted_degree = frozen_degree
        bin_info.chi2_red = chi2_red
        bin_info.tau_F = tau_F
        bin_info.tau_M = tau_M
        bin_info.tau_B = tau_B

        final_n_eff = compute_n_eff(kernel_weights, sigma_eff)
        diagnostics.n_eff = final_n_eff

        if logger:
            if final_n_eff < n_eff_warning_threshold:
                logger.warning(
                    f"E={bin_info.energy_mev:.4f} MeV: Low N_eff={final_n_eff:.1f} "
                    f"(threshold: {n_eff_warning_threshold})"
                )
            if diagnostics.weight_span_ratio > weight_span_warning_ratio:
                logger.warning(
                    f"E={bin_info.energy_mev:.4f} MeV: Wide weight span "
                    f"({diagnostics.weight_span_95:.4f} MeV = {diagnostics.weight_span_ratio:.1f}×σE)"
                )

        results.append(NominalFitResult(
            energy_mev=bin_info.energy_mev,
            energy_index=bin_info.index,
            exfor_df=exfor_df,
            experiments_info=experiments_info,
            kernel_weights=kernel_weights,
            frozen_degree=frozen_degree,
            nominal_coeffs=nominal_coeffs,
            sigma_eff=sigma_eff,
            tau_info=tau_info if tau_info else {'tau_F': 0.0, 'tau_M': 0.0, 'tau_B': 0.0},
            chi2_red=chi2_red,
            has_data=True,
            kernel_diagnostics=diagnostics,
            degree_weights=degree_weights,
            all_degrees_info=all_degrees_info,
        ))

        # Log comprehensive energy bin information
        if logger:
            # Energy header with bin boundaries or σE depending on method
            if experiment_selection_method == "energy_bin":
                logger.info(
                    f"E = {bin_info.energy_mev:.4f} MeV (bin: [{bin_info.bin_lower_mev:.4f}, {bin_info.bin_upper_mev:.4f}] MeV):"
                )
            else:
                logger.info(
                    f"E = {bin_info.energy_mev:.4f} MeV (σE = {bin_info.sigma_E_mev:.4f} MeV):"
                )

            # Experiments used (condensed - one line per experiment with ranges)
            condensed_lines = _format_condensed_experiments(experiments_info)
            for line in condensed_lines:
                logger.info(line)

            # Fit results
            logger.info(
                f"  Fit: L={frozen_degree}, χ²/dof={chi2_red:.2f}, {len(exfor_df)} pts, N_eff={final_n_eff:.1f}"
            )
            logger.info(
                f"  τ values: τ_F={tau_F:.4f}, τ_M={tau_M:.4f}, τ_B={tau_B:.4f}"
            )
            logger.info("")  # Blank line between bins

    if tau_smoothing_window > 1 and use_band_discrepancy:
        tau_by_energy = {r.energy_mev: r.tau_info for r in results if r.has_data}
        if len(tau_by_energy) >= tau_smoothing_window:
            smoothed_tau = smooth_tau_in_energy(tau_by_energy, window=tau_smoothing_window)
            for r in results:
                if r.has_data and r.energy_mev in smoothed_tau:
                    r.tau_info = smoothed_tau[r.energy_mev]

    return results


# Import PrecomputedEnergyData, _precompute_energy_data, _sample_one_realization, run_mc_per_realization
# from original - these don't need changes
# For brevity, we reference them from the original implementation
# In practice, these would be copied here

# For the migration, we'll import the main run function logic


def load_exfor_with_new_api(
    exfor_directory: str = None,
    db_path: str = None,
    source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    projectile: str = "N",
    mt: int = None,
    energy_range: tuple = None,
    supplementary_json_files: List[str] = None,
    exclude_experiments: Optional[List[str]] = None,
    logger=None,
):
    """
    Load EXFOR data using the new kika.exfor module API.

    Supports multiple data sources: JSON files, X4Pro database, or automatic
    fallback (database with JSON fallback for missing entries).

    Returns data in the legacy format for compatibility with existing code.

    Parameters
    ----------
    exfor_directory : str, optional
        Path to EXFOR data directory (for JSON source or fallback)
    db_path : str, optional
        Path to X4Pro database. Uses KIKA_X4PRO_DB_PATH env var if None.
    source : str, optional
        Data source: "json", "database", "auto" (default), or "both"
    target_zaid : int or List[int], optional
        Target ZAID(s) for database queries. Can be:
        - Single ZAID (e.g., 26056 for Fe-56)
        - List of ZAIDs (e.g., [26056, 26000] for Fe-56 + natural iron)
    projectile : str, optional
        Projectile for database queries (default: "N")
    mt : int, optional
        ENDF MT number for database queries
    energy_range : tuple, optional
        (min, max) energy range in MeV for filtering
    supplementary_json_files : List[str], optional
        List of additional JSON file paths to load (for experiments not in database)
    logger : optional
        Logger instance

    Returns
    -------
    exfor_cache : Dict[float, List[Tuple[pd.DataFrame, Dict]]]
        Legacy format data cache
    sorted_energies : List[float]
        Sorted list of available energies
    """
    if logger:
        logger.info(f"  Using NEW kika.exfor module (read_all_exfor)")
        logger.info(f"  Data source: {source}")
        if source in ("database", "auto", "both"):
            logger.info(f"  Database path: {db_path or 'default (env var or builtin)'}")
            if target_zaid:
                if isinstance(target_zaid, list):
                    logger.info(f"  Target ZAIDs: {target_zaid}")
                else:
                    logger.info(f"  Target ZAID: {target_zaid}")
        if supplementary_json_files:
            logger.info(f"  Supplementary JSON files: {len(supplementary_json_files)}")
            for f in supplementary_json_files:
                logger.info(f"    - {f}")

    # Load with new API - get all objects by identifier
    exfor_dict, load_status = read_all_exfor(
        directory=exfor_directory,
        group_by_energy=False,
        source=source,
        db_path=db_path,
        target_zaid=target_zaid,
        projectile=projectile,
        mt=mt,
        energy_range=energy_range,
        supplementary_json_files=supplementary_json_files,
        exclude_experiments=exclude_experiments,
        return_load_status=True,
    )

    # Extract list of ExforAngularDistribution objects
    exfor_objects = list(exfor_dict.values())

    if logger:
        logger.info(f"  Loaded {len(exfor_objects)} EXFOR datasets")

        # Log supplementary file load results
        if supplementary_json_files:
            logger.info("  Supplementary file load results:")
            for item in load_status.get('loaded', []):
                logger.info(f"    LOADED: {item['id']} ({item.get('label', 'unknown')}, {item['n_energies']} energies)")
            for item in load_status.get('skipped', []):
                logger.warning(f"    SKIPPED: {item['id']} - {item['reason']}")
            for item in load_status.get('failed', []):
                logger.warning(f"    FAILED: {item['file']} - {item['error']}")

    # Convert to legacy format using build_exfor_cache_from_objects
    exfor_cache, sorted_energies = build_exfor_cache_from_objects(
        exfor_objects,
        exclude_experiments=exclude_experiments,
    )

    return exfor_cache, sorted_energies


def run_exfor_to_endf_sampling_v2(
    endf_file: str,
    exfor_directory: str = None,
    output_dir: str = None,
    n_samples: int = 10,
    energy_min_mev: float = 1.0,
    energy_max_mev: float = 3.0,
    mt_number: int = 2,
    max_degree: int = 8,
    select_degree: Optional[str] = "aicc",
    ridge_lambda: float = 0.0,
    m_proj_u: float = 1.008665,
    m_targ_u: float = 55.93494,
    delta_t_ns: float = 10.0,
    flight_path_m: float = 27.037,
    n_sigma_cutoff: float = 3.0,
    use_band_discrepancy: bool = True,
    min_points_per_band: int = 6,
    max_tau_fraction: float = 0.25,
    tau_smoothing_window: int = 3,
    sigma_norm: float = 0.05,
    use_model_averaging: bool = True,
    min_degree_for_averaging: int = 3,
    min_kernel_weight_fraction: float = 1e-3,
    max_experiment_weight_fraction: float = 0.5,
    n_eff_warning_threshold: float = 5.0,
    weight_span_warning_ratio: float = 3.0,
    experiment_selection_method: str = "global_convolution",
    tikhonov_lambda: float = 0.001,
    n_procs: int = 1,
    base_seed: int = 42,
    generate_nominal_endf: bool = True,
    generate_mc_mean_endf: bool = True,
    generate_samples_endf: bool = True,
    generate_covariance: bool = True,
    generate_mf34: bool = False,
    # Database configuration (new parameters)
    exfor_db_path: str = None,
    exfor_source: str = "auto",
    target_zaid: Union[int, List[int]] = None,
    target_projectile: str = "N",
    supplementary_json_files: List[str] = None,
    # Experiment exclusion and uncertainty floor
    exclude_experiments: Optional[List[str]] = None,
    min_relative_uncertainty: float = 0.0,
):
    """
    Main function to generate ENDF samples from EXFOR angular distribution data.

    This is the v2 version using the new kika.exfor module API.

    CHANGES FROM v1:
    - Uses read_all_exfor() from kika.exfor instead of load_all_exfor_data()
    - Uses create_mf34_from_covariance and write_mf34_to_file from kika.endf.writers
    - Supports X4Pro database backend in addition to JSON files

    Parameters
    ----------
    endf_file : str
        Path to reference ENDF file
    exfor_directory : str, optional
        Path to EXFOR JSON files directory
    output_dir : str
        Output directory for generated files
    exfor_db_path : str, optional
        Path to X4Pro database. Uses KIKA_X4PRO_DB_PATH env var if None.
    exfor_source : str, optional
        Data source: "json", "database", "auto" (default), or "both"
    target_zaid : int, optional
        Target ZAID for database queries (e.g., 26056 for Fe-56)
    target_projectile : str, optional
        Projectile for database queries (default: "N")
    (other parameters documented inline)
    """
    global _logger

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_path / f'exfor_to_endf_{timestamp}.log'
    _logger = DualLogger(str(log_file))
    _set_logger(_logger)

    separator = "=" * 80
    _logger.info(separator)
    _logger.info("EXFOR-to-ENDF Angular Distribution Sampling (v2 - using kika.exfor)")
    _logger.info(separator)
    _logger.info(f"Timestamp: {datetime.now().isoformat()}")
    _logger.info("")

    print(f"[INFO] Starting EXFOR-to-ENDF sampling (v2)")
    print(f"[INFO] Log file: {log_file}")

    # Log comprehensive methodology
    _logger.info("[METHODOLOGY]")
    _logger.info("")
    _logger.info("  API Version:")
    _logger.info("    - EXFOR Loading: kika.exfor.read_all_exfor (NEW)")
    _logger.info("    - Frame transforms: kika.exfor.transforms")
    _logger.info("    - MF34 creation: kika.endf.writers.create_mf34_from_covariance (NEW)")
    _logger.info("")

    # Experiment selection method documentation
    _logger.info("  Experiment Selection Method:")
    if experiment_selection_method == "global_convolution":
        _logger.info("    - Method: GLOBAL CONVOLUTION (Tikhonov regularization)")
        _logger.info("      Fits ALL energy points simultaneously using a global optimization")
        _logger.info("      that properly accounts for energy resolution smearing.")
        _logger.info(f"    - Tikhonov λ: {tikhonov_lambda}")
        _logger.info(f"    - Energy kernel: Gaussian with σE from TOF resolution")
    elif experiment_selection_method == "kernel_weights":
        _logger.info("    - Method: KERNEL WEIGHTS (Gaussian weighting)")
        _logger.info("      Each ENDF energy fitted independently with EXFOR points weighted")
        _logger.info("      by Gaussian kernel: g_ij = exp(-0.5 * ((E_i - E_j)/σE)²)")
        _logger.info(f"    - Kernel cutoff: ±{n_sigma_cutoff}σ")
        _logger.info(f"    - Min weight fraction: {min_kernel_weight_fraction}")
        _logger.info(f"    - Max experiment weight fraction: {max_experiment_weight_fraction}")
    else:  # energy_bin
        _logger.info("    - Method: ENERGY BIN MATCHING")
        _logger.info("      Experiments selected if their energy falls within bin boundaries.")
        _logger.info("      Bin boundaries: midpoints between ENDF grid energies.")
        _logger.info("      Weights: uniform (all points within bin get weight = 1.0)")
    _logger.info("")

    # TOF resolution parameters
    if experiment_selection_method in ("kernel_weights", "global_convolution"):
        _logger.info("  TOF Energy Resolution Parameters:")
        _logger.info(f"    - Time resolution (Δt): {delta_t_ns} ns")
        _logger.info(f"    - Flight path (L): {flight_path_m} m")
        _logger.info(f"    - σE calculation: σE = 2E × (Δt/t) where t = L/v")
        _logger.info("")

    # Angular-band discrepancy model
    if use_band_discrepancy:
        _logger.info("  Angular-Band Discrepancy Model:")
        _logger.info("    Used to account for systematic differences between experiments")
        _logger.info("    that vary with scattering angle region:")
        _logger.info("    - Forward band (τ_F):  μ > 0.5  (θ < 60°)")
        _logger.info("    - Mid band (τ_M):      |μ| ≤ 0.5 (60° ≤ θ ≤ 120°)")
        _logger.info("    - Backward band (τ_B): μ < -0.5 (θ > 120°)")
        _logger.info(f"    - Min points per band: {min_points_per_band}")
        _logger.info(f"    - Max τ fraction: {max_tau_fraction} (25% cap)")
        _logger.info(f"    - τ smoothing window: {tau_smoothing_window} energy bins")
        _logger.info("")

    # Legendre fitting
    _logger.info("  Legendre Polynomial Fitting:")
    _logger.info(f"    - Maximum degree: {max_degree}")
    _logger.info(f"    - Degree selection: {select_degree if select_degree else 'fixed (use max)'}")
    _logger.info(f"    - Ridge regularization λ: {ridge_lambda}")
    if use_model_averaging:
        _logger.info(f"    - Model averaging: enabled (min degree: {min_degree_for_averaging})")
    else:
        _logger.info("    - Model averaging: disabled")
    _logger.info("")

    # Warning legend
    _logger.info("[WARNING LEGEND]")
    _logger.info("  The following warnings may appear during fitting:")
    _logger.info("")
    _logger.info("  - 'No EXFOR data': No experimental points found within energy tolerance")
    _logger.info("      -> Effect: Coefficients will be INTERPOLATED from neighboring bins")
    _logger.info("      -> Action: Check data coverage for this energy region")
    _logger.info("")
    _logger.info("  - 'Low N_eff=X.X (threshold: Y)': Effective sample size is low")
    _logger.info("      -> Cause: Few independent data points or highly unequal weights")
    _logger.info("      -> Effect: Larger statistical uncertainty in fitted coefficients")
    _logger.info("      -> Action: Consider widening σE cutoff or checking data availability")
    _logger.info("")
    _logger.info("  - 'Wide weight span (X.XXXX MeV = Y.Yxσ_E)': Energy spread of contributing data")
    _logger.info("      -> Cause: EXFOR data energies span a wide range around target energy")
    _logger.info("      -> Effect: Potential bias from energy-dependent shape changes")
    _logger.info("      -> Action: Review if energy resolution assumptions are appropriate")
    _logger.info("")
    if experiment_selection_method == "kernel_weights":
        _logger.info("  - 'Experiment capping applied': One experiment dominated the weights")
        _logger.info(f"      -> Cause: Experiment exceeded {max_experiment_weight_fraction*100:.0f}% of total weight")
        _logger.info("      -> Effect: Weight redistributed to prevent single-experiment bias")
        _logger.info("      -> Action: This is protective; review if capping is frequent")
        _logger.info("")

    _logger.info("  - 'INTERPOLATED from neighboring bins': No EXFOR data at this energy")
    _logger.info("      -> Cause: No experimental data available within energy tolerance")
    _logger.info("      -> Effect: Coefficients linearly interpolated from neighboring bins with data")
    _logger.info("      -> Note: Original ENDF coefficients are NEVER used (ensures independence)")
    _logger.info("")
    _logger.info("  - 'Below/Above EXFOR data range - extrapolating': Energy outside data coverage")
    _logger.info("      -> Cause: Energy bin is below/above the range of available EXFOR data")
    _logger.info("      -> Effect: Uses nearest neighbor's coefficients (no interpolation possible)")
    _logger.info("      -> Action: Expand energy range coverage or accept extrapolation uncertainty")
    _logger.info("")

    # Fixed parameters
    _logger.info("[FIXED PARAMETERS]")
    _logger.info(f"  ENDF file: {endf_file}")
    _logger.info(f"  Output directory: {output_dir}")
    _logger.info(f"  Energy range: [{energy_min_mev:.3f}, {energy_max_mev:.3f}] MeV")
    _logger.info(f"  MT number: {mt_number}")
    _logger.info(f"  Target mass (m_targ): {m_targ_u} u")
    _logger.info(f"  Projectile mass (m_proj): {m_proj_u} u")
    _logger.info(f"  MC samples: {n_samples}")
    _logger.info(f"  Parallel processes: {n_procs}")
    _logger.info(f"  Base seed: {base_seed}")
    if exclude_experiments:
        _logger.info(f"  Excluded experiments: {exclude_experiments}")
    else:
        _logger.info(f"  Excluded experiments: None")
    if min_relative_uncertainty > 0:
        _logger.info(f"  Min relative uncertainty floor: {min_relative_uncertainty*100:.1f}%")
    else:
        _logger.info(f"  Min relative uncertainty floor: Disabled")
    _logger.info("")
    _logger.info(separator)

    # Validate inputs
    if not os.path.exists(endf_file):
        _logger.error(f"ENDF file not found: {endf_file}", console=True)
        return

    if not os.path.isdir(exfor_directory):
        _logger.error(f"EXFOR directory not found: {exfor_directory}", console=True)
        return

    # Step 1: Pre-load EXFOR data (using NEW API with database support)
    _logger.info("")
    _logger.info("[STEP 1] Pre-loading EXFOR data using NEW kika.exfor module")
    _logger.info(f"  Source: {exfor_source}")
    if exfor_source in ("database", "auto", "both"):
        _logger.info(f"  Database: {exfor_db_path or 'default'}")
    if exfor_directory:
        _logger.info(f"  JSON directory: {exfor_directory}")

    print(f"[INFO] Pre-loading EXFOR data (source={exfor_source})")
    t_exfor_start = time.time()

    try:
        exfor_cache, sorted_exfor_energies = load_exfor_with_new_api(
            exfor_directory=exfor_directory,
            db_path=exfor_db_path,
            source=exfor_source,
            target_zaid=target_zaid,
            projectile=target_projectile,
            mt=mt_number,
            energy_range=(energy_min_mev, energy_max_mev) if energy_min_mev and energy_max_mev else None,
            supplementary_json_files=supplementary_json_files,
            exclude_experiments=exclude_experiments,
            logger=_logger,
        )
        t_exfor_elapsed = time.time() - t_exfor_start

        n_exfor_files = sum(len(entries) for entries in exfor_cache.values())
        _logger.info(f"  Loaded {n_exfor_files} EXFOR experiments at {len(sorted_exfor_energies)} unique energies")
        _logger.info(f"  EXFOR energy range: [{min(sorted_exfor_energies):.4f}, {max(sorted_exfor_energies):.4f}] MeV")
        _logger.info(f"  Pre-loading completed in {t_exfor_elapsed:.2f} seconds")
        print(f"[INFO] Loaded {n_exfor_files} EXFOR experiments in {t_exfor_elapsed:.1f}s")
    except Exception as e:
        _logger.error(f"Failed to load EXFOR data: {str(e)}", console=True)
        return

    # Step 2: Read ENDF and extract energy grid
    _logger.info("")
    _logger.info("[STEP 2] Reading ENDF file and extracting energy grid")

    try:
        endf = read_endf(endf_file)
        mf4 = endf.get_file(4)

        if mf4 is None:
            _logger.error("MF4 section not found in ENDF file", console=True)
            return

        mt_data = mf4.sections.get(mt_number)
        if mt_data is None:
            _logger.error(f"MT{mt_number} not found in MF4", console=True)
            return

        if not isinstance(mt_data, (MF4MTLegendre, MF4MTMixed)):
            _logger.error(f"MT{mt_number} is not Legendre or Mixed type (LTT={mt_data._ltt})", console=True)
            return

        energies_ev = np.array(mt_data.legendre_energies)
        original_coeffs = mt_data.legendre_coefficients

        _logger.info(f"  Found {len(energies_ev)} energy points in MF4/MT{mt_number}")

    except Exception as e:
        _logger.error(f"Failed to read ENDF file: {str(e)}", console=True)
        return

    # Step 3: Compute energy bins
    _logger.info("")
    _logger.info("[STEP 3] Computing energy bins with TOF-based resolution")

    energy_bins = compute_energy_bins_with_tof_resolution(
        energies_ev=energies_ev,
        energy_min_mev=energy_min_mev,
        energy_max_mev=energy_max_mev,
        delta_t_ns=delta_t_ns,
        flight_path_m=flight_path_m,
    )

    if not energy_bins:
        _logger.error(f"No energy points in range [{energy_min_mev}, {energy_max_mev}] MeV", console=True)
        return

    for bin_info in energy_bins:
        if bin_info.index < len(original_coeffs):
            bin_info.original_coeffs = list(original_coeffs[bin_info.index])

    _logger.info(f"  Processing {len(energy_bins)} energy bins")
    print(f"[INFO] Processing {len(energy_bins)} energy bins")

    # Step 4: Nominal fits
    _logger.info("")
    _logger.info("[STEP 4] Phase 1: Nominal fits")

    t_fit_start = time.time()

    nominal_results = perform_nominal_fits(
        energy_bins=energy_bins,
        exfor_cache=exfor_cache,
        sorted_energies=sorted_exfor_energies,
        n_sigma=n_sigma_cutoff,
        max_degree=max_degree,
        select_degree=select_degree,
        ridge_lambda=ridge_lambda,
        m_proj_u=m_proj_u,
        m_targ_u=m_targ_u,
        use_band_discrepancy=use_band_discrepancy,
        min_points_per_band=min_points_per_band,
        max_tau_fraction=max_tau_fraction,
        tau_smoothing_window=tau_smoothing_window,
        min_kernel_weight_fraction=min_kernel_weight_fraction,
        max_experiment_weight_fraction=max_experiment_weight_fraction,
        n_eff_warning_threshold=n_eff_warning_threshold,
        weight_span_warning_ratio=weight_span_warning_ratio,
        experiment_selection_method=experiment_selection_method,
        min_degree_for_averaging=min_degree_for_averaging,
        delta_t_ns=delta_t_ns,
        flight_path_m=flight_path_m,
        tikhonov_lambda=tikhonov_lambda,
        exclude_experiments=exclude_experiments,
        min_relative_uncertainty=min_relative_uncertainty,
        logger=_logger,
    )

    t_nominal_elapsed = time.time() - t_fit_start
    n_with_data = sum(1 for nr in nominal_results if nr.has_data)
    _logger.info(f"  Nominal fits completed in {t_nominal_elapsed:.2f}s")
    _logger.info(f"  Bins with EXFOR data: {n_with_data}/{len(nominal_results)}")
    print(f"[INFO] Nominal fits completed ({n_with_data}/{len(nominal_results)} with data)")

    # Step 4b: Interpolate missing bins (NEVER use original ENDF coefficients)
    n_missing = len(nominal_results) - n_with_data
    if n_missing > 0:
        _logger.info("")
        _logger.info("[STEP 4b] Interpolating missing energy bins")
        _logger.info(f"  Bins needing interpolation: {n_missing}")

        nominal_results = interpolate_missing_nominal_fits(
            nominal_results=nominal_results,
            logger=_logger,
        )

        # Count results after interpolation
        n_with_data_after = sum(1 for nr in nominal_results if nr.has_data)
        n_interpolated = sum(1 for nr in nominal_results if nr.interpolated)
        _logger.info(f"  After interpolation: {n_with_data_after}/{len(nominal_results)} bins have coefficients")
        _logger.info(f"  ({n_interpolated} interpolated, {n_with_data_after - n_interpolated} from EXFOR)")
        _logger.info(f"  IMPORTANT: Original ENDF coefficients are NEVER used as fallback")
        _logger.info(f"  (This ensures an independent evaluation based solely on EXFOR data)")
        print(f"[INFO] Interpolated {n_interpolated} bins without EXFOR data")

    # Log experiment summary (quick overview of all data sources used)
    log_experiments_summary(nominal_results, logger=_logger)

    # Step 5: MC sampling
    # Perform actual Monte Carlo sampling for bins with EXFOR data.
    # Interpolated bins use nominal coefficients (no data to sample from).
    _logger.info("")
    _logger.info("[STEP 5] Phase 2: MC sampling")
    _logger.info(f"  Generating {n_samples} MC samples per energy bin")

    # Initialize all_samples: Dict[sample_idx, Dict[energy_idx, coeffs]]
    all_samples = {s_idx: {} for s_idx in range(n_samples)}

    n_sampled = 0
    n_interpolated_used = 0

    for nr in nominal_results:
        if not nr.has_data:
            continue

        energy_idx = nr.energy_index

        if nr.interpolated:
            # Interpolated bins: use nominal coefficients for all samples
            # (no EXFOR data to sample from)
            endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
            for s_idx in range(n_samples):
                all_samples[s_idx][energy_idx] = endf_coeffs
            n_interpolated_used += 1
        else:
            # Bins with EXFOR data: perform actual MC sampling
            # Call sample_legendre_coefficients with n_samples > 1 to generate diverse samples
            try:
                coef_df, _ = sample_legendre_coefficients(
                    nr.exfor_df,
                    value_col="value",
                    unc_col="unc",
                    degree=nr.frozen_degree,  # Use the degree from nominal fit
                    max_degree=max_degree,
                    select_degree=None,  # Don't re-select, use frozen degree
                    ridge_lambda=ridge_lambda,
                    external_weights=nr.kernel_weights if len(nr.kernel_weights) > 0 else None,
                    n_samples=n_samples,  # Actual MC sampling!
                    use_band_discrepancy=use_band_discrepancy,
                    min_points_per_band=min_points_per_band,
                    max_tau_fraction=max_tau_fraction,
                )

                # Store each sample's coefficients
                for s_idx in range(n_samples):
                    sample_coeffs = coef_df.iloc[s_idx].to_numpy()
                    endf_coeffs = endf_normalize_legendre_coeffs(sample_coeffs, include_a0=False)
                    all_samples[s_idx][energy_idx] = endf_coeffs

                n_sampled += 1

            except Exception as e:
                # Fallback to nominal coefficients if sampling fails
                _logger.warning(f"  MC sampling failed for E={nr.energy_mev:.4f} MeV: {e}")
                endf_coeffs = endf_normalize_legendre_coeffs(nr.nominal_coeffs, include_a0=False)
                for s_idx in range(n_samples):
                    all_samples[s_idx][energy_idx] = endf_coeffs

    _logger.info(f"  MC sampled: {n_sampled} bins, interpolated: {n_interpolated_used} bins")

    # Step 6: Save coefficients
    _logger.info("")
    _logger.info("[STEP 6] Saving Legendre coefficients")

    try:
        npz_file, csv_file = save_all_legendre_coefficients(
            nominal_results=nominal_results,
            all_samples=all_samples,
            output_dir=str(output_path),
            max_degree=max_degree,
        )
        _logger.info(f"  Saved to: {csv_file}")
    except Exception as e:
        _logger.error(f"Failed to save coefficients: {str(e)}", console=True)
        npz_file, csv_file = None, None

    # Step 7: Covariance
    cov_matrix = None
    energy_indices = [nr.energy_index for nr in nominal_results if nr.has_data]

    if generate_covariance:
        _logger.info("")
        _logger.info("[STEP 7] Computing covariance matrix")

        cov_matrix, corr_matrix, param_labels = compute_covariance_from_samples(
            all_samples=all_samples,
            energy_indices=energy_indices,
            max_order=max_degree,
        )

        # Validate covariance: check that diagonal values are non-trivial
        diag = np.diag(cov_matrix)
        diag_nonzero = diag[diag > 0]
        if len(diag_nonzero) > 0:
            min_diag = np.min(diag_nonzero)
            max_diag = np.max(diag_nonzero)
            mean_diag = np.mean(diag_nonzero)
            # Compute standard deviation of normalized Legendre coefficients
            # Note: The coefficients a_l = (c_l/c0)/(2l+1) are already normalized
            # by the total cross section, making them dimensionless. Their std dev
            # is thus inherently a fractional quantity (not a "relative uncertainty"
            # in the traditional sense of std/mean).
            mean_coeff_std = np.sqrt(mean_diag)
            _logger.info(f"  Diagonal stats: min={min_diag:.2e}, max={max_diag:.2e}, mean={mean_diag:.2e}")
            _logger.info(f"  Mean Legendre coeff std: {mean_coeff_std:.4f}")
            _logger.info(f"  (As fraction of unity: {mean_coeff_std*100:.2f}% - coeffs are normalized)")

            if np.all(diag < 1e-20):
                _logger.error(
                    "  WARNING: All diagonal covariance values are essentially zero!",
                    console=True
                )
                _logger.error(
                    "  This indicates MC sampling failed - all samples may be identical.",
                    console=True
                )
        else:
            _logger.warning("  WARNING: No positive diagonal elements in covariance matrix!")

        np.save(output_path / "legendre_covariance.npy", cov_matrix)
        np.save(output_path / "legendre_correlation.npy", corr_matrix)
        _logger.info(f"  Covariance matrix shape: {cov_matrix.shape}")

    # Step 8: Write ENDF files
    evaluation_file = None
    if generate_mc_mean_endf:
        _logger.info("")
        _logger.info("[STEP 8] Writing evaluation ENDF file")

        try:
            evaluation_file = write_evaluation_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                all_samples=all_samples,
                output_dir=str(output_path),
            )
            _logger.info(f"  Evaluation ENDF: {evaluation_file}")
        except Exception as e:
            _logger.error(f"Failed to write evaluation ENDF: {str(e)}", console=True)

    nominal_file = None
    if generate_nominal_endf:
        _logger.info("")
        _logger.info("[STEP 8b] Writing nominal ENDF file")

        try:
            nominal_file = write_nominal_endf(
                original_endf_file=endf_file,
                mt_number=mt_number,
                nominal_results=nominal_results,
                output_dir=str(output_path),
            )
            _logger.info(f"  Nominal ENDF: {nominal_file}")
        except Exception as e:
            _logger.error(f"Failed to write nominal ENDF: {str(e)}", console=True)

    # Step 9: Write sample files
    output_files = []
    if generate_samples_endf:
        _logger.info("")
        _logger.info("[STEP 9] Writing ENDF sample files")

        output_files = write_endf_samples_batch(
            original_endf_file=endf_file,
            mt_number=mt_number,
            all_samples=all_samples,
            output_dir=str(output_path),
        )
        _logger.info(f"  Written {len(output_files)} sample files")

    # Step 10: MF34 (using library functions)
    if generate_mf34 and generate_covariance and cov_matrix is not None:
        _logger.info("")
        _logger.info("[STEP 10] Writing MF34 using kika.endf.writers")

        try:
            endf_orig = read_endf(endf_file)
            mf1 = endf_orig.get_file(1)
            mt451 = mf1.sections.get(451) if mf1 else None

            if mt451:
                za = mt451._za
                awr = mt451._awr
                mat = mt451._mat
            else:
                za = 26056.0
                awr = 55.845
                mat = 2631

            mf4 = endf_orig.get_file(4)
            mt_data = mf4.sections.get(mt_number)
            all_energies_ev = np.array(mt_data.legendre_energies)

            processed_energies_ev = np.array([all_energies_ev[i] for i in energy_indices])
            if energy_indices[-1] + 1 < len(all_energies_ev):
                energy_grid_ev = np.append(processed_energies_ev, all_energies_ev[energy_indices[-1] + 1])
            else:
                if len(processed_energies_ev) > 1:
                    delta = processed_energies_ev[-1] - processed_energies_ev[-2]
                    energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] + delta)
                else:
                    energy_grid_ev = np.append(processed_energies_ev, processed_energies_ev[-1] * 1.1)

            # Use LIBRARY function (new location)
            mf34 = create_mf34_from_covariance(
                cov_matrix=cov_matrix,
                energy_grid_ev=energy_grid_ev,
                max_order=max_degree,
                za=za,
                awr=awr,
                mat=mat,
                mt=mt_number,
            )

            # Use LIBRARY function (new location)
            if evaluation_file:
                write_mf34_to_file(evaluation_file, mf34, evaluation_file)
                _logger.info(f"  MF34 added to evaluation: {evaluation_file}")

            if nominal_file:
                write_mf34_to_file(nominal_file, mf34, nominal_file)
                _logger.info(f"  MF34 added to nominal: {nominal_file}")

        except Exception as e:
            _logger.error(f"Failed to write MF34: {str(e)}", console=True)

    # Summary
    total_time = time.time() - t_exfor_start
    _logger.info("")
    _logger.info(separator)
    _logger.info("[SUMMARY]")
    _logger.info(f"  Total execution time: {total_time:.2f}s")
    _logger.info(f"  API Version: kika.exfor (v2)")
    _logger.info(separator)

    print(f"\n[INFO] Completed! Output directory: {output_path}")
    print(f"[INFO] Total time: {total_time:.1f}s")

    return nominal_results, all_samples, output_files


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    run_exfor_to_endf_sampling_v2(
        endf_file=ENDF_FILE,
        exfor_directory=EXFOR_DIRECTORY,
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        energy_min_mev=ENERGY_MIN_MEV,
        energy_max_mev=ENERGY_MAX_MEV,
        mt_number=MT_NUMBER,
        max_degree=MAX_LEGENDRE_DEGREE,
        select_degree=SELECT_DEGREE,
        ridge_lambda=RIDGE_LAMBDA,
        m_proj_u=M_PROJ_U,
        m_targ_u=M_TARG_U,
        delta_t_ns=DELTA_T_NS,
        flight_path_m=FLIGHT_PATH_M,
        n_sigma_cutoff=N_SIGMA_CUTOFF,
        use_band_discrepancy=USE_BAND_DISCREPANCY,
        min_points_per_band=MIN_POINTS_PER_BAND,
        max_tau_fraction=MAX_TAU_FRACTION,
        tau_smoothing_window=TAU_SMOOTHING_WINDOW,
        sigma_norm=NORMALIZATION_SIGMA,
        use_model_averaging=USE_MODEL_AVERAGING,
        min_degree_for_averaging=MIN_DEGREE_FOR_AVERAGING,
        min_kernel_weight_fraction=MIN_KERNEL_WEIGHT_FRACTION,
        max_experiment_weight_fraction=MAX_EXPERIMENT_WEIGHT_FRACTION,
        n_eff_warning_threshold=N_EFF_WARNING_THRESHOLD,
        weight_span_warning_ratio=WEIGHT_SPAN_WARNING_RATIO,
        experiment_selection_method=EXPERIMENT_SELECTION_METHOD,
        n_procs=N_PROCS,
        base_seed=BASE_SEED,
        generate_nominal_endf=GENERATE_NOMINAL_ENDF,
        generate_mc_mean_endf=GENERATE_MC_MEAN_ENDF,
        generate_samples_endf=GENERATE_SAMPLES_ENDF,
        generate_covariance=GENERATE_COVARIANCE,
        generate_mf34=GENERATE_MF34,
        # Database configuration
        exfor_db_path=EXFOR_DB_PATH,
        exfor_source=EXFOR_SOURCE,
        target_zaid=TARGET_ZAIDS,
        target_projectile=TARGET_PROJECTILE,
        supplementary_json_files=SUPPLEMENTARY_JSON_FILES,
        # Experiment exclusion and uncertainty floor
        exclude_experiments=EXCLUDE_EXPERIMENTS,
        min_relative_uncertainty=MIN_RELATIVE_UNCERTAINTY,
    )
