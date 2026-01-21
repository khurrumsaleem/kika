# EXFOR-to-ENDF Angular Distribution Sampling

This document explains the implementation of the `exfor_to_endf_sampling_v2.py` script, which generates sampled ENDF files from EXFOR experimental angular distribution data.

## Overview

The script performs the following workflow:

1. Load EXFOR experimental data (from database and/or JSON files)
2. Read reference ENDF file and extract the MF4 Legendre energy grid
3. For each ENDF energy point, fit Legendre polynomials to nearby EXFOR data
4. Generate N Monte Carlo samples of the coefficients
5. Write output ENDF files with sampled coefficients

---

## Experiment Selection Methods

Three methods are available for selecting and weighting EXFOR data:

### 1. Global Convolution (`global_convolution`) - RECOMMENDED

**What it does:**
- Fits ALL ENDF energy points simultaneously in a single large linear system
- Each EXFOR measurement contributes to multiple ENDF energies according to its resolution-weighted probability
- Uses Tikhonov regularization to enforce smooth energy dependence

**Mathematical Model:**

For a measurement at nominal energy $E_j$ and angle $\mu_i$:

$$
y_{ij} \approx \sum_k w_{jk} \left( \sum_\ell c_\ell(E_k) P_\ell(\mu_i) \right)
$$

where:
- $c_\ell(E_k)$ are Legendre coefficients at ENDF grid energy $E_k$
- $w_{jk} = \Phi\left(\frac{E_{k,\text{high}} - E_j}{\sigma_j}\right) - \Phi\left(\frac{E_{k,\text{low}} - E_j}{\sigma_j}\right)$ is the probability that the true energy lies in bin $k$
- $P_\ell(\mu_i)$ is the Legendre polynomial of degree $\ell$

**Tikhonov Regularization:**

Enforces smooth energy dependence via second-difference penalty:
$$
R = \lambda \sum_\ell \sum_k \left( c_\ell(E_{k+1}) - 2c_\ell(E_k) + c_\ell(E_{k-1}) \right)^2
$$

**When to use:** Default choice. Best for production runs.

**Key parameter:** `GLOBAL_CONV_LAMBDA` (default: 0.001)

### 2. Kernel Weights (`kernel_weights`)

**What it does:**
- Fits each ENDF energy point independently
- Uses Gaussian kernel weighting based on energy resolution:
  $$
  g_{ij} = \exp\left(-0.5 \left(\frac{E_i - E_j}{\sigma E_j}\right)^2\right)
  $$

**When to use:** Debugging, or when you want independent fits at each energy.

**Key parameters:**
- `N_SIGMA_CUTOFF`: Energy window in units of σE (default: 3.0)
- `MIN_KERNEL_WEIGHT_FRACTION`: Minimum weight threshold (default: 0.001)

### 3. Energy Bin (`energy_bin`)

**What it does:**
- Simple hard binning without resolution-based weighting
- All data within bin boundaries get equal weight

**When to use:** Quick tests, or when energy resolution effects are negligible.

---

## Configuration Parameters

### Input/Output Configuration

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `ENDF_FILE` | Path to reference ENDF file | `/path/to/26-Fe-56g.txt` |
| `EXFOR_DIRECTORY` | Path to EXFOR JSON files (for fallback) | `/path/to/exfor/data/` |
| `OUTPUT_DIR` | Output directory for generated files | `/path/to/output/` |
| `N_SAMPLES` | Number of Monte Carlo samples | 10-1000 |

### Database Configuration

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `EXFOR_DB_PATH` | Path to X4Pro database (None = use env var) | `None` |
| `EXFOR_SOURCE` | Data source: `"database"`, `"json"`, `"auto"`, `"both"` | `"database"` |
| `SUPPLEMENTARY_JSON_FILES` | List of extra JSON files to load | `["path/to/27673002.json"]` |
| `TARGET_ZAID` | Target ZAID for database queries | `26056` (Fe-56) |
| `TARGET_PROJECTILE` | Projectile type | `"N"` (neutrons) |

### Energy Range

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `ENERGY_MIN_MEV` | Minimum energy in MeV | `1.0` |
| `ENERGY_MAX_MEV` | Maximum energy in MeV | `3.0` |
| `MT_NUMBER` | Reaction type (MT=2 for elastic) | `2` |

### TOF Energy Resolution Parameters

These parameters control how energy resolution is computed for TOF experiments.

| Parameter | Description | Formula | Typical Value |
|-----------|-------------|---------|---------------|
| `DELTA_T_NS` | Time resolution in nanoseconds | - | `10.0` |
| `FLIGHT_PATH_M` | Flight path in meters | - | `27.037` |
| `N_SIGMA_CUTOFF` | Gaussian kernel cutoff | ±N_SIGMA × σE | `3.0` |

**Energy Resolution Formula:**
$$
\sigma E = E \times 2 \times \Delta t \times \frac{\sqrt{2E/m_n}}{L}
$$

where:
- $E$ is the neutron energy
- $\Delta t$ is the time resolution (`DELTA_T_NS`)
- $L$ is the flight path (`FLIGHT_PATH_M`)
- $m_n$ is the neutron mass

**Note:** These are DEFAULT values. The script can also read experiment-specific values from:
1. JSON files (in `method.energy_resolution_input`)
2. TOF metadata file (`kika/exfor/tof_metadata.json`)

### Target Isotope Masses

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `M_PROJ_U` | Projectile mass in atomic mass units | `1.008665` (neutron) |
| `M_TARG_U` | Target mass in atomic mass units | `55.93494` (Fe-56) |

These are used for LAB→CM frame conversion when EXFOR data is in the LAB frame.

### Legendre Fitting Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MAX_LEGENDRE_DEGREE` | Maximum Legendre polynomial order | `8` |
| `SELECT_DEGREE` | Degree selection: `"aicc"`, `"bic"`, or `None` | `"aicc"` |
| `RIDGE_LAMBDA` | Ridge regularization parameter | `0.0` |

**Degree Selection Methods:**
- `"aicc"`: Corrected Akaike Information Criterion (preferred for small samples)
- `"bic"`: Bayesian Information Criterion (more conservative, prefers simpler models)
- `None`: Use `MAX_LEGENDRE_DEGREE` fixed

### Angular-Band Discrepancy Parameters

These control the uncertainty inflation model that replaces global Birge scaling.

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `USE_BAND_DISCREPANCY` | Enable band-based uncertainty model | `True` |
| `MIN_POINTS_PER_BAND` | Minimum points to estimate τ per band | `6` |
| `MAX_TAU_FRACTION` | Cap τ at this fraction of cross section | `0.25` |
| `TAU_SMOOTHING_WINDOW` | Moving median window for τ(E) smoothing | `3` |

**Angular Bands:**
- Forward: $\mu > 0.5$ ($\theta < 60°$)
- Mid: $|\mu| \leq 0.5$ ($60° \leq \theta \leq 120°$)
- Backward: $\mu < -0.5$ ($\theta > 120°$)

**How it works:**

1. Compute normalized residuals in each band: $r_i = (y_i - y_{\text{fit},i}) / \sigma_i$
2. Compute robust scale: $s_b = 1.4826 \times \text{median}(|r - \text{median}(r)|)$
3. If $s_b > 1$: $\tau_b = \text{median}(\sigma_b) \times \sqrt{s_b^2 - 1}$
4. Effective uncertainty: $\sigma^2_{\text{eff}} = \sigma^2 + \tau^2_b$

This captures systematic differences between experiments that vary with angle.

### Kernel Weight Control

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MIN_KERNEL_WEIGHT_FRACTION` | Drop points with weight < max × this | `0.001` |
| `MAX_EXPERIMENT_WEIGHT_FRACTION` | Cap any experiment's weight fraction | `0.5` |
| `N_EFF_WARNING_THRESHOLD` | Warn if N_eff < this | `5.0` |
| `WEIGHT_SPAN_WARNING_RATIO` | Warn if max/min weight > this | `3.0` |

**N_eff (Effective Sample Size):**
$$
N_{\text{eff}} = \frac{\left(\sum w_i\right)^2}{\sum w_i^2}
$$
A low $N_{\text{eff}}$ indicates that a few experiments dominate the fit.

### Global Convolution Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `GLOBAL_CONV_LAMBDA` | Tikhonov regularization strength | `0.001` |

**Choosing λ:**
- Larger $\lambda$ → smoother coefficients across energy, but may over-smooth real structure
- Smaller $\lambda$ → follows data more closely, but may have unphysical oscillations
- Start with 0.001 and adjust based on residuals

### Model Averaging Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `USE_MODEL_AVERAGING` | Average over multiple Legendre degrees | `True` |
| `MIN_DEGREE_FOR_AVERAGING` | Minimum degree to include | `1` |

When enabled, the final coefficients are a weighted average over different polynomial orders, with weights based on AICc/BIC.

### Normalization Uncertainty

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `NORMALIZATION_SIGMA` | Per-experiment normalization uncertainty | `0.05` (5%) |

Accounts for systematic normalization differences between experiments.

### Output Generation Options

| Parameter | Description |
|-----------|-------------|
| `GENERATE_NOMINAL_ENDF` | Write ENDF with nominal (best-fit) coefficients |
| `GENERATE_MC_MEAN_ENDF` | Write ENDF with mean of MC samples |
| `GENERATE_SAMPLES_ENDF` | Write N individual sample ENDF files |
| `GENERATE_COVARIANCE` | Write covariance matrix files |
| `GENERATE_MF34` | Write MF34 covariance section |

### Parallelization

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `N_PROCS` | Number of parallel processes | `5` |
| `BASE_SEED` | Random seed for reproducibility | `42` |

---

## Workflow Details

### Step 1: Load EXFOR Data

Data is loaded from the X4Pro database and/or JSON files. The `supplementary_json_files` parameter allows loading experiments not yet in the database.

TOF parameters (flight path, time resolution) are obtained from:
1. The experiment's JSON file (if loaded from JSON)
2. The [tof_metadata.json](../kika/exfor/tof_metadata.json) file (for database entries)
3. Default values (`FLIGHT_PATH_M`, `DELTA_T_NS`)

### Step 2: Compute Energy Bins

For each ENDF energy point, compute:
- Bin boundaries (midpoints between grid energies)
- Energy resolution $\sigma E$ using TOF formula
- Energy window for data selection ($\pm N_{\sigma} \times \sigma E$)

### Step 3: Fit Legendre Coefficients

**Global Convolution Method:**
1. Build sparse design matrix connecting all EXFOR points to all ENDF energies
2. Weight each contribution by resolution overlap probability
3. Add Tikhonov regularization for energy smoothness
4. Solve using sparse least squares (LSQR)

**Per-Energy Methods:**
1. Select EXFOR data within energy window
2. Apply kernel weights (Gaussian or overlap)
3. Transform LAB→CM if needed
4. Fit Legendre polynomial using weighted least squares
5. Select optimal degree using AICc/BIC
6. Apply angular-band discrepancy model

### Step 4: Generate Samples

1. For each energy, compute coefficient covariance matrix
2. Generate N samples from multivariate normal distribution
3. Normalize each sample to preserve cross section integral

### Step 5: Write Output

- Nominal ENDF: Best-fit coefficients
- MC Mean ENDF: Average of all samples
- Sample ENDFs: Individual realizations
- Covariance files: Coefficient covariances per energy
- MF34: ENDF covariance format

---

## Tips for Parameter Tuning

### Energy Resolution Sensitivity

If your fits show unexpected oscillations:
- Check that `FLIGHT_PATH_M` and `DELTA_T_NS` are reasonable for your experiments
- Try increasing `GLOBAL_CONV_LAMBDA` for smoother results
- Verify experiment-specific TOF parameters in [tof_metadata.json](../kika/exfor/tof_metadata.json)

### Handling Sparse Data

If some energies have little data:
- The global convolution method handles this automatically via regularization
- Check `N_eff` warnings in the log
- Consider increasing `N_SIGMA_CUTOFF` to include more data

### Uncertainty Estimation

If uncertainties seem too small or too large:
- Check `USE_BAND_DISCREPANCY` setting
- Adjust `MAX_TAU_FRACTION` to limit uncertainty inflation
- Review `TAU_SMOOTHING_WINDOW` for energy smoothness of τ

### Performance

For faster runs:
- Use `EXPERIMENT_SELECTION_METHOD = "energy_bin"` for quick tests
- Reduce `N_SAMPLES` during development
- Increase `N_PROCS` for parallel processing

---

## File Outputs

| File Pattern | Description |
|--------------|-------------|
| `Fe56_nominal.txt` | ENDF with best-fit coefficients |
| `Fe56_mc_mean.txt` | ENDF with MC mean coefficients |
| `Fe56_sample_001.txt` | Individual MC sample |
| `legendre_coeffs_all.csv` | All coefficients by energy |
| `covariance_E*.csv` | Covariance matrices per energy |
| `run.log` | Detailed execution log |

---

## Example Configuration

```python
# Basic setup for Fe-56 elastic scattering
ENDF_FILE = "/path/to/26-Fe-56g.txt"
EXFOR_SOURCE = "database"
TARGET_ZAID = 26056
MT_NUMBER = 2

# Energy range
ENERGY_MIN_MEV = 1.0
ENERGY_MAX_MEV = 3.0

# Recommended method
EXPERIMENT_SELECTION_METHOD = "global_convolution"
GLOBAL_CONV_LAMBDA = 0.001

# Fitting
MAX_LEGENDRE_DEGREE = 8
SELECT_DEGREE = "aicc"
USE_BAND_DISCREPANCY = True

# Output
N_SAMPLES = 100
GENERATE_NOMINAL_ENDF = True
GENERATE_SAMPLES_ENDF = True
```
