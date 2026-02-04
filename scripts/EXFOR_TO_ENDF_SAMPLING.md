# EXFOR-to-ENDF Angular Distribution Sampling

This document provides comprehensive documentation for the `exfor_to_endf_sampling_v2.py` script, which generates sampled ENDF files from EXFOR experimental angular distribution data.

**Version:** v2 (uses `kika.exfor` module for EXFOR data handling)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quick Start Guide](#2-quick-start-guide)
3. [Experiment Selection Methods](#3-experiment-selection-methods)
4. [Configuration Reference](#4-configuration-reference)
5. [Parameter Dependency Matrix](#5-parameter-dependency-matrix)
6. [Practical Guidance](#6-practical-guidance)
7. [Example Configurations](#7-example-configurations)
8. [Output Files](#8-output-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Technical Reference](#10-technical-reference)

---

## 1. Overview

### Workflow Summary

The script performs the following workflow:

1. **Load EXFOR experimental data** from database and/or JSON files
2. **Read reference ENDF file** and extract the MF4 Legendre energy grid
3. **Compute energy bins** with TOF-based resolution for each ENDF energy point
4. **Fit Legendre polynomials** to EXFOR data using one of three methods
5. **Generate N Monte Carlo samples** of the coefficients
6. **Write output ENDF files** with sampled coefficients and covariance data

### Key Features

| Feature Category | Improvements |
|-----------------|--------------|
| **Experiment Weighting** | Per-experiment normalization (1.1), Weight guards (1.2, 3.2) |
| **Shape Fitting** | FREEZE_C0 for shape-only fits (1.3), Two-pass shape-only mode (3.4) |
| **Energy Correlation** | Energy jitter MC for cross-bin coupling (1.4) |
| **Global Fitting** | L-dependent regularization (3.3), Resolution-aware convolution |

---

## 2. Quick Start Guide

### Minimal Configuration

```python
# Essential paths
ENDF_FILE = "/path/to/26-Fe-56g.txt"
EXFOR_SOURCE = "database"
OUTPUT_DIR = "/path/to/output/"

# Target nucleus
TARGET_ZAIDS = [26056]  # Fe-56
MT_NUMBER = 2           # Elastic scattering

# Energy range
ENERGY_MIN_MEV = 1.0
ENERGY_MAX_MEV = 3.0

# Default method
EXPERIMENT_SELECTION_METHOD = "energy_bin"

# Output options
N_SAMPLES = 100
GENERATE_NOMINAL_ENDF = True
GENERATE_COVARIANCE = True
```

### Recommended Production Configuration (energy_bin)

```python
# === RECOMMENDED SETTINGS FOR PRODUCTION (energy_bin) ===

# Method selection (default)
EXPERIMENT_SELECTION_METHOD = "energy_bin"

# Shape-only fitting and weighting
FREEZE_C0 = True                  # Shape-only refits
NORMALIZE_BY_N_POINTS = True      # Equal weight per experiment
USE_ENERGY_JITTER = True          # Cross-bin energy correlations

# Fitting parameters
MAX_LEGENDRE_DEGREE = 8
SELECT_DEGREE = "aicc"
USE_BAND_DISCREPANCY = True
MIN_RELATIVE_UNCERTAINTY = 0.05   # 5% uncertainty floor

# Output
N_SAMPLES = 500
GENERATE_MF34 = True
N_PROCS = 8
```

### Alternative Production Configuration (global_convolution)

```python
# === PRODUCTION SETTINGS FOR global_convolution ===
# Prefer this method when energy resolution smearing is important
# or data is sparse across the energy range.

EXPERIMENT_SELECTION_METHOD = "global_convolution"

# Global convolution tuning
GLOBAL_CONV_LAMBDA = 0.001        # Regularization strength
L_DEPENDENT_POWER = 2.0           # Higher-order smoothing
SKIP_C0_REGULARIZATION = True     # Don't penalize c0 variations
GLOBAL_CONV_SHAPE_ONLY = True     # Two-pass shape-only fit

# Fitting parameters
MAX_LEGENDRE_DEGREE = 8
SELECT_DEGREE = "aicc"
MIN_RELATIVE_UNCERTAINTY = 0.05   # 5% uncertainty floor

# Output
N_SAMPLES = 500
GENERATE_MF34 = True
N_PROCS = 8
```

### Common Starting Points

| Use Case | Method | Key Settings |
|----------|--------|--------------|
| Production (default) | `energy_bin` | `FREEZE_C0=True`, `USE_ENERGY_JITTER=True`, `N_SAMPLES=500` |
| Production (sparse data) | `global_convolution` | `GLOBAL_CONV_SHAPE_ONLY=True`, `GLOBAL_CONV_LAMBDA=0.01` |
| Quick validation | `energy_bin` | `N_SAMPLES=25`, `N_PROCS=1` |
| Debug single energy | `kernel_weights` | `DEDUPE_NOMINAL=True`, `DEDUPE_MC=False` |
| Dense data regions | `energy_bin` | `NORMALIZE_BY_N_POINTS=True` |

---

## 3. Experiment Selection Methods

Three methods are available for selecting and weighting EXFOR data. Each has different characteristics suited to different use cases.

### Method Comparison Table

| Feature | `global_convolution` | `kernel_weights` | `energy_bin` |
|---------|---------------------|------------------|--------------|
| **Fitting approach** | All energies simultaneously | Per-energy independent | Per-energy independent |
| **Energy resolution** | Resolution-aware convolution | Gaussian kernel weighting | Hard bin boundaries |
| **Regularization** | Tikhonov smoothing | None | None |
| **Speed** | Slowest (but parallel) | Medium | Fastest |
| **Handles sparse data** | Excellent (via regularization) | Good | Poor |
| **Energy correlations** | Naturally included | Via two-pass dedupe | Via energy jitter |
| **Shape-only mode** | Two-pass (Improvement 3.4) | Not available | FREEZE_C0 (Improvement 1.3) |
| **Recommended for** | Sparse data, resolution-critical | Debugging, comparisons | General use (default) |

---

### 3.1 Global Convolution (`global_convolution`)

Fits ALL ENDF energy points simultaneously in a single large linear system. Each EXFOR measurement contributes to multiple ENDF energies according to its resolution-weighted probability.

#### Mathematical Model

For a measurement at nominal energy $E_j$ and angle $\mu_i$:

$$
y_{ij} \approx \sum_k w_{jk} \left( \sum_\ell c_\ell(E_k) P_\ell(\mu_i) \right)
$$

where:
- $c_\ell(E_k)$ are Legendre coefficients at ENDF grid energy $E_k$
- $w_{jk} = \Phi\left(\frac{E_{k,\text{high}} - E_j}{\sigma_j}\right) - \Phi\left(\frac{E_{k,\text{low}} - E_j}{\sigma_j}\right)$ is the probability that the true energy lies in bin $k$
- $P_\ell(\mu_i)$ is the Legendre polynomial of degree $\ell$
- $\Phi$ is the standard normal CDF

#### Tikhonov Regularization

Enforces smooth energy dependence via second-difference penalty:

$$
R = \lambda \sum_\ell (1 + \ell)^p \sum_k \left( c_\ell(E_{k+1}) - 2c_\ell(E_k) + c_\ell(E_{k-1}) \right)^2
$$

where:
- $\lambda$ = `GLOBAL_CONV_LAMBDA` (regularization strength)
- $p$ = `L_DEPENDENT_POWER` (higher-order suppression)
- The $(1+\ell)^p$ factor applies stronger smoothing to higher Legendre orders

#### Shape-Only Mode (Improvement 3.4)

When `GLOBAL_CONV_SHAPE_ONLY = True`:

1. **First pass**: Solve full system for all coefficients
2. **Freeze c0**: Extract $c_0(E_k)$ from first pass for each energy
3. **Second pass**: Re-solve with $c_0$ fixed, only fitting $c_1, c_2, \ldots$

This prevents normalization differences between experiments from affecting shape coefficients.

#### Weight Guards (Improvement 3.2)

When energy resolution causes data to be truncated at bin edges:

| Weight Sum | Action |
|------------|--------|
| $\geq 0.95$ | Normal processing |
| $0.5 - 0.95$ | Warning logged, data included |
| $< 0.5$ | Dataset skipped entirely |

Configure via `MIN_WEIGHT_SUM_THRESHOLD` (default: 0.95).

#### L-Dependent Regularization (Improvement 3.3)

Higher Legendre orders receive stronger smoothing:

| `L_DEPENDENT_POWER` | Effect |
|---------------------|--------|
| 0.0 | Equal regularization for all $\ell$ |
| 2.0 | Moderate suppression of high-$\ell$ oscillations (recommended) |
| 4.0 | Strong suppression, very smooth high-$\ell$ behavior |

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GLOBAL_CONV_LAMBDA` | 0.001 | Tikhonov regularization strength |
| `L_DEPENDENT_POWER` | 2.0 | L-scaling exponent for regularization |
| `SKIP_C0_REGULARIZATION` | True | Don't smooth $c_0$ across energy |
| `MIN_WEIGHT_SUM_THRESHOLD` | 0.95 | Minimum acceptable weight sum |
| `GLOBAL_CONV_SHAPE_ONLY` | True | Enable two-pass shape-only fitting |

---

### 3.2 Kernel Weights (`kernel_weights`)

Fits each ENDF energy point independently using Gaussian kernel weighting based on energy resolution.

#### Mathematical Model

EXFOR data are weighted by:

$$
g_{ij} = \exp\left(-\frac{1}{2} \left(\frac{E_i - E_j}{\sigma E_j}\right)^2\right)
$$

where:
- $E_i$ = ENDF grid energy
- $E_j$ = EXFOR measurement energy
- $\sigma E_j$ = experiment-specific energy resolution

#### Two-Pass Deduplication

Controls how experiments with multiple energies in range are handled:

| Setting | `DEDUPE_NOMINAL` | `DEDUPE_MC` | Use Case |
|---------|------------------|-------------|----------|
| Stable nominal | True | True | Standard operation |
| Energy correlations | True | False | Enables cross-bin correlations in MC |
| All data | False | False | Maximum data usage (may cause instability) |

- **DEDUPE_NOMINAL=True**: For nominal fits, select only the closest energy from each experiment
- **DEDUPE_MC=False**: For MC sampling, use all energies to capture energy correlations

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_KERNEL_WEIGHT_FRACTION` | 0.001 | Drop points with weight < max × this |
| `MAX_EXPERIMENT_WEIGHT_FRACTION` | 0.5 | Cap any experiment's weight fraction |
| `N_EFF_WARNING_THRESHOLD` | 5.0 | Warn if $N_{\text{eff}} <$ this |
| `WEIGHT_SPAN_WARNING_RATIO` | 3.0 | Warn if energy span > ratio × $\sigma E$ |
| `DEDUPE_NOMINAL` | True | Deduplicate for nominal fits |
| `DEDUPE_MC` | False | Deduplicate for MC sampling |

---

### 3.3 Energy Bin (`energy_bin`)

Simple energy binning with hard boundaries. All data within bin boundaries receive equal weight (after per-experiment normalization).

#### Per-Experiment Weighting (Improvement 1.1)

When `NORMALIZE_BY_N_POINTS = True`:

$$
w_{\text{point}} = \frac{1}{n_{\text{points}}}
$$

This ensures each experiment contributes equally regardless of angular sampling density.

#### Weight Guard (Improvement 1.2)

Prevents single experiments from dominating:

| `MAX_EXP_WEIGHT_FRAC_BIN` | Effect |
|---------------------------|--------|
| 1.0 | No capping (disabled) |
| 0.5 | Cap each experiment at 50% of total weight |
| 0.25 | Cap at 25% for highly redundant data |

#### Shape-Only Fitting (Improvement 1.3)

When `FREEZE_C0 = True`:

1. First fit: Determine $c_0$ from full weighted least squares
2. Second fit: Fix $c_0$, fit only shape coefficients $c_1, c_2, \ldots$

This is analogous to `GLOBAL_CONV_SHAPE_ONLY` but for per-energy fits.

#### Energy Jitter for Cross-Bin Coupling (Improvement 1.4)

When `USE_ENERGY_JITTER = True`:

For each MC sample:
1. Perturb each dataset's energy: $E' = E + \epsilon \cdot \sigma_E$, where $\epsilon \sim N(0,1)$ clipped at $\pm$ `JITTER_N_SIGMA_CLIP`
2. Reassign data to bins based on perturbed energy
3. Refit coefficients with new bin assignments

This introduces cross-bin correlations in the MC samples that reflect energy resolution uncertainty.

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NORMALIZE_BY_N_POINTS` | True | Equal weight per experiment |
| `MAX_EXP_WEIGHT_FRAC_BIN` | 0.5 | Per-experiment weight cap |
| `FREEZE_C0` | True | Fix $c_0$ for shape-only fits |
| `USE_ENERGY_JITTER` | True | Enable energy jitter MC |
| `JITTER_N_SIGMA_CLIP` | 3.0 | Clip jitter at ±n sigma |
| `TRACK_BIN_JUMPS` | True | Log bin jump statistics |

---

## 4. Configuration Reference

### 4.1 Input/Output Paths

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ENDF_FILE` | str | (required) | Path to reference ENDF file with MF4 data |
| `EXFOR_DIRECTORY` | str | (optional) | Path to EXFOR JSON files for `source="json"` or `"auto"` |
| `EXFOR_DB_PATH` | str | None | Path to X4Pro SQLite database. None = use `KIKA_X4PRO_DB_PATH` env variable |
| `OUTPUT_DIR` | str | (required) | Directory for all generated output files |

### 4.2 Data Source Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EXFOR_SOURCE` | str | `"database"` | Data source: `"database"`, `"json"`, `"auto"`, or `"both"` |
| `TARGET_ZAIDS` | List[int] | `[26056]` | Target ZAID(s) for database queries. Include natural isotope (e.g., 26000) for more data |
| `TARGET_PROJECTILE` | str | `"N"` | Projectile type: `"N"` (neutrons), `"P"` (protons), etc. |
| `SUPPLEMENTARY_JSON_FILES` | List[str] | `[]` | Additional JSON files to load regardless of source setting |

**When to change:**
- Use `TARGET_ZAIDS = [26056, 26000]` to include both Fe-56 and natural iron data
- Use `"auto"` for `EXFOR_SOURCE` to get database + JSON fallback for missing experiments

### 4.3 Output Generation Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GENERATE_NOMINAL_ENDF` | bool | True | Write ENDF with best-fit (nominal) coefficients |
| `GENERATE_MC_MEAN_ENDF` | bool | True | Write ENDF with MC mean coefficients |
| `GENERATE_SAMPLES_ENDF` | bool | False | Write individual MC sample ENDF files |
| `GENERATE_COVARIANCE` | bool | True | Write covariance matrix files (.npy, .csv) |
| `GENERATE_MF34` | bool | True | Write MF34 covariance section in ENDF format |
| `N_SAMPLES` | int | 25 | Number of Monte Carlo samples to generate |

**When to change:**
- Set `GENERATE_SAMPLES_ENDF = True` for TMC (Total Monte Carlo) applications
- Increase `N_SAMPLES` to 500-1000 for production; use 25-50 for development
- Disable `GENERATE_MF34` if only raw covariance matrices are needed

### 4.3b Multigroup Covariance Options

These parameters control the adaptive multigroup collapse of fine-grid covariance matrices into broader energy groups suitable for MF34 output.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GENERATE_MULTIGROUP_COVARIANCE` | bool | True | Enable adaptive multigroup collapse of the fine-grid covariance |
| `MULTIGROUP_RHO_MIN` | float | 0.90 | Minimum Pearson correlation between adjacent energy points to allow merging into one group. Range 0.85-0.95 |
| `MULTIGROUP_SIGMA_RATIO_MAX` | float | 2.0 | Maximum ratio of largest to smallest standard deviation within a group. Range 1.5-2.0 |
| `MULTIGROUP_MIN_WIDTH_FACTOR` | float | 2.0 | Minimum group width as a multiple of the local median energy resolution $\sigma_E$. Prevents groups narrower than the physical resolution |
| `MF34_COVARIANCE_TYPE` | str | `"both"` | Which covariance grids to write in MF34: `"fine"` (original per-ENDF-point), `"multigroup"` (collapsed groups), or `"both"` |
| `MULTIGROUP_VARIANCE_PERCENTILE` | float | 90.0 | Percentile of fine-grid variances within each group used for the diagonal of the multigroup covariance. 50 = median, 90 = conservative, 100 = maximum |

**How multigroup collapse works:**

The algorithm merges adjacent fine-grid energy points into broader groups while preserving the covariance structure. Two adjacent points are candidates for merging if:

1. Their Pearson correlation $\rho_{ij} \geq$ `MULTIGROUP_RHO_MIN` — they must be highly correlated (behaving similarly under sampling)
2. Their standard-deviation ratio $\max(\sigma_i, \sigma_j) / \min(\sigma_i, \sigma_j) \leq$ `MULTIGROUP_SIGMA_RATIO_MAX` — uncertainties must be comparable
3. The resulting group width $\geq$ `MULTIGROUP_MIN_WIDTH_FACTOR` $\times$ median($\sigma_E$) — the group must not be narrower than the experimental resolution

The `MULTIGROUP_VARIANCE_PERCENTILE` controls how the diagonal (variance) of the collapsed matrix is computed. A value of 90 means the group variance is set to the 90th percentile of fine-grid variances within that group, which is conservative without being driven by outliers.

**When to change:**
- Lower `MULTIGROUP_RHO_MIN` to 0.85 for coarser grouping (fewer groups)
- Raise to 0.95 for finer grouping that only merges very similar points
- Set `MF34_COVARIANCE_TYPE = "fine"` to skip multigroup output entirely
- Increase `MULTIGROUP_VARIANCE_PERCENTILE` to 100 for the most conservative variance estimates

### 4.4 General Parameters (All Methods)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ENERGY_MIN_MEV` | float | 1.0 | Minimum energy to process (MeV) |
| `ENERGY_MAX_MEV` | float | 3.0 | Maximum energy to process (MeV) |
| `MT_NUMBER` | int | 2 | Reaction MT number (2 = elastic scattering) |
| `M_PROJ_U` | float | 1.008665 | Projectile mass in atomic mass units |
| `M_TARG_U` | float | 55.93494 | Target mass in atomic mass units (Fe-56) |
| `MAX_LEGENDRE_DEGREE` | int | 8 | Maximum Legendre polynomial order |
| `SELECT_DEGREE` | str/None | `"aicc"` | Degree selection: `"aicc"`, `"bic"`, or `None` (use max) |
| `RIDGE_LAMBDA` | float | 1e-6 | Ridge regularization for Legendre fitting |
| `RIDGE_POWER` | int | 4 | Power exponent for L-dependent ridge penalty. Higher values suppress high-order Legendre coefficients more strongly. Used in formula: λ × (ℓ+1)^RIDGE_POWER |
| `DF_METHOD` | str | `"hat"` | Method for computing effective degrees of freedom in AICc/BIC. `"hat"` uses trace of hat matrix (more accurate for regularized fits), `"naive"` uses number of parameters |
| `N_PROCS` | int | 5 | Number of parallel processes (1 = sequential) |
| `BASE_SEED` | int | 42 | Random seed for reproducibility |

**Degree Selection Methods (`SELECT_DEGREE`):**

Information criteria balance goodness-of-fit against model complexity to prevent overfitting. Both AICc and BIC are computed for each candidate degree $k$ and the degree with the lowest value is selected.

- **AICc** (corrected Akaike Information Criterion):

$$
\text{AIC} = n \ln(\text{RSS}/n) + 2k, \quad \text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

The correction term $2k(k+1)/(n-k-1)$ is critical when $n/k < 40$, which is common for angular distributions with few data points. Without the correction, AIC tends to select overly complex models in small samples.

- **BIC** (Bayesian Information Criterion):

$$
\text{BIC} = n \ln(\text{RSS}/n) + k \ln(n)
$$

BIC replaces the $2k$ penalty with $k\ln(n)$, which grows with sample size. For $n \geq 8$, $\ln(n) > 2$, so BIC penalizes complexity more heavily than AIC and prefers simpler models. BIC is asymptotically consistent (selects the true model order as $n \to \infty$), while AICc minimizes prediction error.

- `None`: Uses `MAX_LEGENDRE_DEGREE` for all fits without model selection. Appropriate when the correct degree is known a priori or when regularization (ridge) already controls complexity.

**Ridge Regularization (`RIDGE_LAMBDA`, `RIDGE_POWER`):**

Ridge regularization prevents ill-conditioned Legendre fits by penalizing large coefficients, especially at high orders. The modified cost function is:

$$
\mathcal{L} = \|\mathbf{y} - \mathbf{A}\mathbf{c}\|^2 + \lambda_{\text{ridge}} \sum_\ell (\ell+1)^p \, c_\ell^2
$$

where $\lambda_{\text{ridge}}$ = `RIDGE_LAMBDA` and $p$ = `RIDGE_POWER`. The $(\ell+1)^p$ factor applies stronger shrinkage to higher-order coefficients, which are more susceptible to noise. With `RIDGE_POWER = 4`, a degree-8 coefficient receives $(9)^4 = 6561$ times more penalty than degree-0.

This is distinct from Tikhonov regularization used in `global_convolution`, which penalizes roughness (second-differences across energy) rather than coefficient magnitude. Ridge regularization operates within a single energy point's fit; Tikhonov operates across energy points.

**Degrees of Freedom (`DF_METHOD`):**

The effective degrees of freedom affect AICc and BIC calculations:

- `"hat"`: Uses $\text{df} = \text{trace}(\mathbf{H})$ where $\mathbf{H} = \mathbf{A}(\mathbf{A}^T\mathbf{W}\mathbf{A} + \boldsymbol{\Lambda})^{-1}\mathbf{A}^T\mathbf{W}$ is the hat matrix. When ridge regularization is active, $\text{df}_{\text{hat}} < k$ because regularization effectively reduces the number of free parameters. This gives more accurate AICc/BIC values for regularized fits.
- `"naive"`: Uses $\text{df} = k$ (number of parameters), ignoring the effect of regularization. Simpler but overestimates complexity for regularized fits.

### 4.5 Experiment Selection Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EXPERIMENT_SELECTION_METHOD` | str | `"energy_bin"` | Method: `"global_convolution"`, `"kernel_weights"`, or `"energy_bin"` |

### 4.6 Experiment Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `EXCLUDE_EXPERIMENTS` | List[str] | `[]` | Experiments to exclude. Formats: `"20743"` (all subentries), `"20743002"`, or `"20743/002"` |
| `MIN_RELATIVE_UNCERTAINTY` | float | 0.05 | Minimum uncertainty floor as fraction (5% = 0.05). Set to 0 to disable |

**Understanding `MIN_RELATIVE_UNCERTAINTY`:**

This parameter applies a floor to quoted experimental uncertainties:

$$
\sigma_{\text{eff}} = \max(\sigma_{\text{quoted}}, \text{MIN\_RELATIVE\_UNCERTAINTY} \times y)
$$

In weighted least squares, each point receives weight $w_i \sim 1/\sigma_i^2$. A measurement with unrealistically small quoted uncertainty (e.g., 0.5%) can receive 100× more weight than a measurement with 5% uncertainty, effectively letting a single experiment determine the fit. Some older experiments quote very small statistical errors without accounting for systematic contributions (detector efficiency, beam normalization, angular acceptance). The floor prevents such points from dominating.

**When to change:**
- Exclude known problematic experiments: `EXCLUDE_EXPERIMENTS = ["20743002"]`
- Increase `MIN_RELATIVE_UNCERTAINTY` to 0.08 if fits are dominated by unrealistically small errors

### 4.7 TOF Energy Resolution (kernel_weights, global_convolution)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DELTA_T_NS` | float | 5.0 | Default time resolution in nanoseconds |
| `FLIGHT_PATH_M` | float | 27.037 | Default flight path in meters |
| `N_SIGMA_CUTOFF` | float | 3.0 | Gaussian kernel cutoff (±n_sigma × $\sigma E$) |

**Physical basis:** In time-of-flight experiments, neutron energy is determined from $E = \frac{1}{2}m_n v^2$ where $v = L/t$ (flight path / flight time). The energy resolution follows from error propagation:

$$
\sigma_E = \left|\frac{dE}{dt}\right| \Delta t = E \times \frac{2\Delta t}{t} = E \times \frac{2\Delta t \cdot v}{L}
$$

Since $v = \sqrt{2E/m_n}$, we get $\sigma_E \propto E^{3/2}$: higher-energy neutrons travel faster, so the same timing uncertainty $\Delta t$ maps to a larger energy spread. At 1 MeV with a 27 m flight path and 5 ns resolution, $\sigma_E \approx 5$ keV. At 3 MeV, $\sigma_E \approx 25$ keV.

These are **default values** when experiment-specific TOF parameters are unavailable. The script preferentially uses:
1. Experiment-specific values from JSON files (`method.energy_resolution_input`)
2. Values from `TOF_PARAMETERS_FILE` (for energy jitter)
3. These defaults as fallback

**Energy Resolution Formula:**

$$
\sigma E = E \times 2 \times \frac{\Delta t}{t} = E \times 2 \times \frac{\Delta t \cdot v}{L}
$$

where $v = c\sqrt{2E/m_n}$ is the neutron velocity.

### 4.8 Global Convolution Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `GLOBAL_CONV_LAMBDA` | float | 0.001 | Tikhonov regularization strength |
| `L_DEPENDENT_POWER` | float | 2.0 | L-scaling exponent (0 = uniform, 2-4 recommended) |
| `SKIP_C0_REGULARIZATION` | bool | True | Don't apply smoothing penalty to $c_0$ |
| `MIN_WEIGHT_SUM_THRESHOLD` | float | 0.95 | Warn if weight_sum < this; skip if < 0.5 |
| `GLOBAL_CONV_SHAPE_ONLY` | bool | True | Enable two-pass shape-only fitting |

**Understanding `SKIP_C0_REGULARIZATION`:**

The coefficient $c_0$ is special: it is proportional to the angle-integrated cross section ($\sigma_{\text{total}} / 4\pi$). Unlike higher-order coefficients that describe angular shape, $c_0$ can genuinely vary rapidly with energy due to resonance structure. The Tikhonov regularization term for $c_0$ would be:

$$
R_{c_0} = \lambda \sum_k \left(c_0(E_{k+1}) - 2c_0(E_k) + c_0(E_{k-1})\right)^2
$$

This second-difference penalty suppresses rapid variation, which would artificially smooth out real resonance structure in the total cross section. Setting `SKIP_C0_REGULARIZATION = True` removes this penalty for $\ell = 0$ while keeping it for $\ell \geq 1$, where smooth energy dependence is a more reasonable prior.

**Understanding `GLOBAL_CONV_SHAPE_ONLY` / `FREEZE_C0`:**

The angular distribution can be written as $\sigma(\mu) = c_0 \left(1 + \sum_{\ell \geq 1} a_\ell P_\ell(\mu)\right)$ where $a_\ell = c_\ell / c_0$ are shape ratios. Different experiments may have different absolute normalizations but agree on the angular shape. Fitting all coefficients simultaneously allows normalization offsets between experiments to bias the shape coefficients.

The two-pass approach (`GLOBAL_CONV_SHAPE_ONLY = True` or `FREEZE_C0 = True`) first determines $c_0$ from a full fit, then freezes it and re-fits only $c_1, c_2, \ldots$. This lets shape coefficients be determined without normalization bias.

**Choosing `GLOBAL_CONV_LAMBDA`:**

| $\lambda$ Value | Effect |
|-----------------|--------|
| 0.0001 | Very weak smoothing, may show oscillations |
| 0.001 | Balanced (default) |
| 0.01 | Strong smoothing, good for sparse data |
| 0.1 | Very strong, may over-smooth real structure |

### 4.9 Kernel Weight Control (kernel_weights only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MIN_KERNEL_WEIGHT_FRACTION` | float | 0.001 | Minimum weight as fraction of max |
| `MAX_EXPERIMENT_WEIGHT_FRACTION` | float | 0.5 | Cap any single experiment's weight |
| `N_EFF_WARNING_THRESHOLD` | float | 5.0 | Warn if $N_{\text{eff}} <$ this |
| `WEIGHT_SPAN_WARNING_RATIO` | float | 3.0 | Warn if span > ratio × $\sigma E$ |
| `DEDUPE_NOMINAL` | bool | True | Deduplicate for nominal fits |
| `DEDUPE_MC` | bool | False | Deduplicate for MC sampling |
| `USE_OVERLAP_WEIGHTS` | bool | True | When True, uses overlap integral weights (probability that true energy lies in each bin). When False, uses simple Gaussian kernel weights. Overlap weights are more accurate for resolution-aware fitting |

**N_eff (Effective Sample Size):**

$$
N_{\text{eff}} = \frac{\left(\sum w_i\right)^2}{\sum w_i^2}
$$

When all experiments have equal weight, $N_{\text{eff}} = N_{\text{experiments}}$. If one experiment has much higher weight (e.g., due to smaller quoted uncertainties or more data points), $N_{\text{eff}}$ drops toward 1, indicating the fit is essentially determined by a single dataset. The threshold `N_EFF_WARNING_THRESHOLD` (default 5.0) flags cases where the effective diversity of contributing experiments is low, which increases the risk of systematic bias from a single measurement.

### 4.10 Angular-Band Discrepancy (kernel_weights, energy_bin)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_BAND_DISCREPANCY` | bool | True | Use band-based uncertainty inflation |
| `MIN_POINTS_PER_BAND` | int | 3 | Minimum points to estimate $\tau$ per band |
| `MAX_TAU_FRACTION` | float | 0.25 | Cap $\tau_b$ at this fraction of cross section |
| `TAU_SMOOTHING_WINDOW` | int | 3 | Moving median window for $\tau(E)$ smoothing |
| `TAU_PRIOR_FLOOR` | bool | True | Apply a prior floor to $\tau_b$ estimates based on well-estimated values from other energy bins |
| `TAU_PRIOR_MIN_EXPERIMENTS` | int | 2 | Minimum number of experiments in a bin for its $\tau$ to count as "well-estimated" |
| `TAU_PRIOR_PERCENTILE` | int | 50 | Percentile of well-estimated $\tau$ values used as the floor (50 = median) |
| `RESCALE_UNC_BY_CHI2` | bool | True | When band discrepancy is disabled, apply Birge scaling (multiply uncertainties by √χ²_red) to account for model-data discrepancy. Ignored when `USE_BAND_DISCREPANCY=True` |
| `ALLOW_SHRINK_UNC` | bool | False | When Birge scaling is applied and χ²_red < 1, allow uncertainties to shrink. When False (default), scaling factor is capped at 1.0 to prevent underestimation of errors |

**Physical motivation:**

When multiple experiments measure the same angular distribution but disagree beyond their quoted uncertainties, this indicates unaccounted systematic errors. Rather than inflating all uncertainties globally, the band approach estimates the discrepancy separately in three angular regions because different systematic effects dominate in different angular ranges:

- **Forward** ($\mu > 0.5$, $\theta < 60°$): Dominated by Coulomb/nuclear interference; sensitive to detector solid-angle corrections
- **Mid** ($|\mu| \leq 0.5$, $60° \leq \theta \leq 120°$): Often most reliable; less affected by forward-peaking or back-angle scattering systematics
- **Backward** ($\mu < -0.5$, $\theta > 120°$): Sensitive to compound-nuclear contributions and multiple scattering corrections

The discrepancy $\tau_b$ is estimated using a robust (MAD-based) scale of the normalized residuals. When experiments agree within their uncertainties, $\tau_b = 0$. When they disagree, the effective uncertainty is inflated: $\sigma_{\text{eff},i}^2 = \sigma_i^2 + \tau_b^2$. See Section 10.2 for the full equations.

**Tau Prior Floor (`TAU_PRIOR_FLOOR`):**

At energy bins with only a single experiment, the inter-experiment discrepancy $\tau_b$ cannot be estimated (there is no disagreement to detect). Setting `TAU_PRIOR_FLOOR = True` applies a floor derived from bins where $\tau$ can be reliably estimated (those with $\geq$ `TAU_PRIOR_MIN_EXPERIMENTS` experiments). The floor is the `TAU_PRIOR_PERCENTILE`-th percentile of those well-estimated $\tau$ values (default: median). This prevents single-experiment bins from having artificially small uncertainties.

**Birge Scaling (`RESCALE_UNC_BY_CHI2`):**

When band discrepancy is disabled, Birge scaling provides a simpler alternative:

$$
\sigma_{\text{scaled}} = \sigma_{\text{original}} \times \max\left(1, \sqrt{\chi^2_{\text{red}}}\right)
$$

When $\chi^2_{\text{red}} > 1$, the data scatter exceeds what the uncertainties predict, suggesting underestimated errors. Birge scaling inflates all uncertainties uniformly to make $\chi^2_{\text{red}} = 1$. Setting `ALLOW_SHRINK_UNC = False` enforces the $\max(1, \ldots)$ to prevent deflating errors when $\chi^2_{\text{red}} < 1$ (which can happen by chance or when the model is over-parameterized).

### 4.11 Per-Experiment Normalization (kernel_weights, energy_bin)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NORMALIZATION_SIGMA` | float | 0.05 | Per-experiment normalization uncertainty (5%) |
| `NORM_DIST` | str | `"lognormal"` | Distribution: `"lognormal"` (always positive) or `"normal"` |

**Physical motivation:**

Different experiments have systematic normalization offsets arising from detector efficiency calibration, beam intensity measurement, target thickness determination, and solid angle uncertainties. These are correlated within each experiment (all angles shift together) but independent between experiments.

In MC sampling, each experiment $k$ receives a random normalization factor:

$$
f_k \sim \text{LogNormal}\left(-\frac{\sigma_n^2}{2}, \sigma_n\right) \quad \text{with } E[f_k] = 1
$$

The log-shift $-\sigma_n^2/2$ ensures unit mean. All data from experiment $k$ are then scaled: $y'_i = f_k \times y_i$. This propagates normalization uncertainty into the MC samples, producing wider coefficient distributions that reflect the true state of knowledge.

**When to change:**
- Increase `NORMALIZATION_SIGMA` to 0.10 if experiments show large systematic offsets
- Use `"normal"` for `NORM_DIST` if negative factors are acceptable (rare)

### 4.12 Model Averaging (kernel_weights, energy_bin)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_MODEL_AVERAGING` | bool | True | Average over multiple Legendre degrees |
| `MIN_DEGREE_FOR_AVERAGING` | int | 1 | Minimum degree to include in averaging |
| `USE_DEGREE_SAMPLING_IN_MC` | bool | True | Sample degree from weight distribution in MC |

**Bayesian Model Averaging:**

Instead of selecting a single best Legendre degree and discarding all alternatives, model averaging computes a weighted combination:

$$
\mathbf{c}_{\text{avg}} = \sum_d w_d \, \mathbf{c}_d, \quad w_d = \frac{\exp(-\frac{1}{2}\Delta\text{AICc}_d)}{\sum_{d'} \exp(-\frac{1}{2}\Delta\text{AICc}_{d'})}
$$

where $\Delta\text{AICc}_d = \text{AICc}_d - \min(\text{AICc})$. Each degree gets a weight based on its relative AICc score. A degree with $\Delta\text{AICc} = 0$ (the best) gets the highest weight, while a degree with $\Delta\text{AICc} = 10$ gets weight $\approx 0.007$ relative to the best.

This accounts for model selection uncertainty: when two degrees have similar AICc scores, neither should be chosen definitively. When `USE_DEGREE_SAMPLING_IN_MC = True`, each MC sample draws its degree from the weight distribution, introducing model selection uncertainty into the sampled coefficients.

### 4.13 Energy Bin Specific (energy_bin only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NORMALIZE_BY_N_POINTS` | bool | True | Equal weight per experiment (1/n_points) |
| `MAX_EXP_WEIGHT_FRAC_BIN` | float | 0.5 | Cap per-experiment weight (1.0 = disabled) |
| `FREEZE_C0` | bool | True | Fix $c_0$ for shape-only refits |

### 4.14 Energy Jitter (energy_bin only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `USE_ENERGY_JITTER` | bool | True | Enable energy jitter for cross-bin correlation |
| `TOF_PARAMETERS_FILE` | str | (path) | JSON file with per-experiment TOF parameters |
| `JITTER_N_SIGMA_CLIP` | float | 3.0 | Clip jitter at ±n_sigma |
| `TRACK_BIN_JUMPS` | bool | True | Track bin jump statistics for diagnostics |

---

## 5. Parameter Dependency Matrix

This matrix shows which parameters apply to which experiment selection methods.

| Parameter | global_convolution | kernel_weights | energy_bin |
|-----------|:------------------:|:--------------:|:----------:|
| **Section 4.1-4.4 (General)** | ✓ | ✓ | ✓ |
| `RIDGE_POWER` | - | ✓ | ✓ |
| `DF_METHOD` | - | ✓ | ✓ |
| **Section 4.6 (Filtering)** | ✓ | ✓ | ✓ |
| `DELTA_T_NS` | ✓ | ✓ | - |
| `FLIGHT_PATH_M` | ✓ | ✓ | - |
| `N_SIGMA_CUTOFF` | ✓ | ✓ | - |
| `GLOBAL_CONV_LAMBDA` | ✓ | - | - |
| `L_DEPENDENT_POWER` | ✓ | - | - |
| `SKIP_C0_REGULARIZATION` | ✓ | - | - |
| `MIN_WEIGHT_SUM_THRESHOLD` | ✓ | - | - |
| `GLOBAL_CONV_SHAPE_ONLY` | ✓ | - | - |
| `MIN_KERNEL_WEIGHT_FRACTION` | ✓ | ✓ | - |
| `MAX_EXPERIMENT_WEIGHT_FRACTION` | - | ✓ | - |
| `N_EFF_WARNING_THRESHOLD` | - | ✓ | - |
| `WEIGHT_SPAN_WARNING_RATIO` | - | ✓ | - |
| `DEDUPE_NOMINAL` | - | ✓ | - |
| `DEDUPE_MC` | - | ✓ | - |
| `USE_OVERLAP_WEIGHTS` | ✓ | ✓ | - |
| `USE_BAND_DISCREPANCY` | - | ✓ | ✓ |
| `MIN_POINTS_PER_BAND` | - | ✓ | ✓ |
| `MAX_TAU_FRACTION` | - | ✓ | ✓ |
| `TAU_SMOOTHING_WINDOW` | - | ✓ | ✓ |
| `RESCALE_UNC_BY_CHI2` | - | ✓ | ✓ |
| `ALLOW_SHRINK_UNC` | - | ✓ | ✓ |
| `NORMALIZATION_SIGMA` | - | ✓ | ✓ |
| `NORM_DIST` | - | ✓ | ✓ |
| `USE_MODEL_AVERAGING` | - | ✓ | ✓ |
| `MIN_DEGREE_FOR_AVERAGING` | - | ✓ | ✓ |
| `USE_DEGREE_SAMPLING_IN_MC` | - | ✓ | ✓ |
| `NORMALIZE_BY_N_POINTS` | - | - | ✓ |
| `MAX_EXP_WEIGHT_FRAC_BIN` | - | - | ✓ |
| `FREEZE_C0` | - | - | ✓ |
| `USE_ENERGY_JITTER` | - | - | ✓ |
| `TOF_PARAMETERS_FILE` | - | - | ✓ |
| `JITTER_N_SIGMA_CLIP` | - | - | ✓ |
| `TRACK_BIN_JUMPS` | - | - | ✓ |
| **Section 4.3b (Multigroup)** | | | |
| `GENERATE_MULTIGROUP_COVARIANCE` | ✓ | ✓ | ✓ |
| `MULTIGROUP_RHO_MIN` | ✓ | ✓ | ✓ |
| `MULTIGROUP_SIGMA_RATIO_MAX` | ✓ | ✓ | ✓ |
| `MULTIGROUP_MIN_WIDTH_FACTOR` | ✓ | ✓ | ✓ |
| `MF34_COVARIANCE_TYPE` | ✓ | ✓ | ✓ |
| `MULTIGROUP_VARIANCE_PERCENTILE` | ✓ | ✓ | ✓ |
| **Tau Prior Floor** | | | |
| `TAU_PRIOR_FLOOR` | - | ✓ | ✓ |
| `TAU_PRIOR_MIN_EXPERIMENTS` | - | ✓ | ✓ |
| `TAU_PRIOR_PERCENTILE` | - | ✓ | ✓ |

---

## 6. Practical Guidance

### 6.1 Choosing a Method

```
                    ┌─────────────────────────────┐
                    │ Is data sparse or energy    │
                    │ resolution critical?        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                   YES                           NO
                    │                             │
                    ▼                             ▼
         ┌──────────────────┐         ┌──────────────────┐
         │ global_convolution│         │ Need detailed    │
         │ (regularized)     │         │ diagnostics?     │
         └──────────────────┘         └────────┬─────────┘
                                                │
                                   ┌────────────┴────────────┐
                                   │                         │
                                   NO                       YES
                                   │                         │
                                   ▼                         ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │ energy_bin       │     │ kernel_weights   │
                        │ (default)        │     │ (debugging)      │
                        └──────────────────┘     └──────────────────┘
```

### 6.2 Parameter Tuning Guide

#### Problem: Fits show unphysical oscillations in energy

| If Method | Try |
|-----------|-----|
| `global_convolution` | Increase `GLOBAL_CONV_LAMBDA` (0.001 → 0.01) |
| `global_convolution` | Increase `L_DEPENDENT_POWER` (2.0 → 3.0) |
| `kernel_weights` | Enable `DEDUPE_NOMINAL = True` |
| Any | Exclude problematic experiments via `EXCLUDE_EXPERIMENTS` |

#### Problem: Uncertainties seem too small

| Symptom | Solution |
|---------|----------|
| $\chi^2_{\text{red}} \gg 1$ | Increase `MIN_RELATIVE_UNCERTAINTY` (0.05 → 0.08) |
| Single experiment dominates | Reduce `MAX_EXPERIMENT_WEIGHT_FRACTION` (0.5 → 0.3) |
| Band discrepancy inactive | Verify `USE_BAND_DISCREPANCY = True` |
| | Reduce `MIN_POINTS_PER_BAND` if needed (default 3) |

#### Problem: Uncertainties seem too large

| Symptom | Solution |
|---------|----------|
| $\tau$ values very high | Reduce `MAX_TAU_FRACTION` (0.25 → 0.15) |
| Normalization inflates errors | Reduce `NORMALIZATION_SIGMA` (0.05 → 0.03) |
| Over-regularized | Reduce `GLOBAL_CONV_LAMBDA` (0.001 → 0.0001) |

#### Problem: Single experiment dominates the fit

| If Method | Solution |
|-----------|----------|
| `kernel_weights` | Set `MAX_EXPERIMENT_WEIGHT_FRACTION = 0.3` |
| `energy_bin` | Set `MAX_EXP_WEIGHT_FRAC_BIN = 0.3` |
| Any | Enable `NORMALIZE_BY_N_POINTS = True` |
| Any | Check `N_EFF_WARNING_THRESHOLD` warnings |

#### Problem: Many bins have no EXFOR data

| If Method | Solution |
|-----------|----------|
| `global_convolution` | Regularization handles this automatically |
| `kernel_weights` | Increase `N_SIGMA_CUTOFF` (3.0 → 4.0) |
| `energy_bin` | Consider switching to `global_convolution` |
| Any | Add natural isotope to `TARGET_ZAIDS` |
| Any | Reduce `ENERGY_MIN_MEV`/`ENERGY_MAX_MEV` range |

### 6.3 Behavior Comparison Tables

#### Effect of Shape-Only Mode

| Setting | Normalization Treatment | When to Use |
|---------|------------------------|-------------|
| `GLOBAL_CONV_SHAPE_ONLY = False` | All coefficients vary together | Well-normalized data |
| `GLOBAL_CONV_SHAPE_ONLY = True` | $c_0$ fixed, only shape varies | Mixed experiment normalizations |
| `FREEZE_C0 = True` (energy_bin) | $c_0$ fixed per-energy | Per-energy shape fits |

#### Effect of Energy Jitter

| `USE_ENERGY_JITTER` | Cross-Bin Correlation | Speed | Use Case |
|---------------------|----------------------|-------|----------|
| False | None | Fast | Independent bins |
| True | Via bin reassignment | Slower | Energy uncertainty propagation |

Interpretation of bin jump diagnostics:
- Jump rate >30%: Grid finer than $\sigma E$, strong energy correlations
- Jump rate 10-30%: Grid comparable to $\sigma E$, moderate correlations
- Jump rate <10%: Grid coarser than $\sigma E$, weak correlations

#### Effect of Normalization Distribution

| `NORM_DIST` | Factor Range | Typical Use |
|-------------|--------------|-------------|
| `"lognormal"` | Always > 0 | Recommended (physical) |
| `"normal"` | Can be negative | Rare edge cases |

---

## 7. Example Configurations

### 7.1 Production: Fe-56 Elastic Scattering

```python
# === PRODUCTION CONFIGURATION ===
# Target: Fe-56 elastic scattering, 1-3 MeV

# Input/Output
ENDF_FILE = "/soft_snc/lib/endf/jeff40/neutrons/26-Fe-56g.txt"
EXFOR_SOURCE = "database"
EXFOR_DB_PATH = "/share_snc/snc/JuanMonleon/EXFOR/x4_iron_angular.db"
OUTPUT_DIR = "/output/Fe56_production/"

# Include natural iron for better coverage
TARGET_ZAIDS = [26056, 26000]
TARGET_PROJECTILE = "N"
MT_NUMBER = 2

# Energy range
ENERGY_MIN_MEV = 1.0
ENERGY_MAX_MEV = 3.0

# Method (global_convolution for resolution-aware fitting)
EXPERIMENT_SELECTION_METHOD = "global_convolution"
GLOBAL_CONV_LAMBDA = 0.001
L_DEPENDENT_POWER = 2.0
SKIP_C0_REGULARIZATION = True
GLOBAL_CONV_SHAPE_ONLY = True
MIN_WEIGHT_SUM_THRESHOLD = 0.95

# Fitting parameters
MAX_LEGENDRE_DEGREE = 8
SELECT_DEGREE = "aicc"
RIDGE_LAMBDA = 1e-6

# Uncertainty handling
MIN_RELATIVE_UNCERTAINTY = 0.05
USE_BAND_DISCREPANCY = True

# Output options
N_SAMPLES = 500
GENERATE_NOMINAL_ENDF = True
GENERATE_MC_MEAN_ENDF = True
GENERATE_SAMPLES_ENDF = False  # Set True for TMC
GENERATE_COVARIANCE = True
GENERATE_MF34 = True

# Performance
N_PROCS = 8
BASE_SEED = 42
```

### 7.2 Development: Quick Validation Run

```python
# === QUICK VALIDATION ===
# Fast testing during development

EXPERIMENT_SELECTION_METHOD = "energy_bin"

# Minimal samples for speed
N_SAMPLES = 10

# Disable heavy outputs
GENERATE_SAMPLES_ENDF = False
GENERATE_MF34 = False

# Single process for easier debugging
N_PROCS = 1

# Narrow energy range
ENERGY_MIN_MEV = 1.5
ENERGY_MAX_MEV = 2.0
```

### 7.3 Debugging: Full Diagnostics

```python
# === DEBUGGING CONFIGURATION ===
# Maximum logging and diagnostics

EXPERIMENT_SELECTION_METHOD = "kernel_weights"

# Enable all diagnostics
DEDUPE_NOMINAL = True
DEDUPE_MC = False  # See all data in MC
TRACK_BIN_JUMPS = True

# Lower thresholds to see more warnings
N_EFF_WARNING_THRESHOLD = 3.0
WEIGHT_SPAN_WARNING_RATIO = 2.0
MIN_WEIGHT_SUM_THRESHOLD = 0.8

# Single process for sequential logging
N_PROCS = 1
N_SAMPLES = 5

# Disable outputs to focus on log
GENERATE_SAMPLES_ENDF = False
```

### 7.4 Energy Bin with Cross-Bin Correlation

```python
# === ENERGY BIN WITH JITTER ===
# Per-energy fitting with energy correlation

EXPERIMENT_SELECTION_METHOD = "energy_bin"

# Per-experiment weighting
NORMALIZE_BY_N_POINTS = True
MAX_EXP_WEIGHT_FRAC_BIN = 0.5

# Shape-only fitting
FREEZE_C0 = True

# Enable energy jitter for cross-bin correlation
USE_ENERGY_JITTER = True
TOF_PARAMETERS_FILE = "/path/to/exfor_tof_parameters.json"
JITTER_N_SIGMA_CLIP = 3.0
TRACK_BIN_JUMPS = True

# Band discrepancy
USE_BAND_DISCREPANCY = True
MIN_POINTS_PER_BAND = 3
MAX_TAU_FRACTION = 0.25
```

### 7.5 Handling Sparse Data Regions

```python
# === SPARSE DATA CONFIGURATION ===
# For energy regions with limited experimental coverage

EXPERIMENT_SELECTION_METHOD = "global_convolution"

# Stronger regularization for sparse data
GLOBAL_CONV_LAMBDA = 0.01  # 10x default
L_DEPENDENT_POWER = 3.0    # Stronger high-L suppression

# More permissive weight threshold
MIN_WEIGHT_SUM_THRESHOLD = 0.7

# Include both specific and natural isotope
TARGET_ZAIDS = [26056, 26000]

# Wider energy window
N_SIGMA_CUTOFF = 4.0  # vs default 3.0
```

---

## 8. Output Files

### 8.1 File Outputs

| File Pattern | Description | Generated When |
|--------------|-------------|----------------|
| `{isotope}_nominal.txt` | ENDF with best-fit Legendre coefficients | `GENERATE_NOMINAL_ENDF = True` |
| `{isotope}_mc_mean.txt` | ENDF with MC mean coefficients | `GENERATE_MC_MEAN_ENDF = True` |
| `{isotope}_sample_{NNN}.txt` | Individual MC sample ENDF files | `GENERATE_SAMPLES_ENDF = True` |
| `legendre_coeffs_all.csv` | All coefficients by energy | Always |
| `legendre_coeffs_samples.csv` | MC samples of coefficients | Always |
| `covariance_matrix.npy` | Full covariance matrix (numpy) | `GENERATE_COVARIANCE = True` |
| `covariance_matrix.csv` | Full covariance matrix (CSV) | `GENERATE_COVARIANCE = True` |
| `{isotope}_mf34.txt` | MF34 covariance section | `GENERATE_MF34 = True` |
| `run.log` | Detailed execution log | Always |

### 8.2 Log File Structure

The `run.log` file contains:

```
=== Run Configuration ===
[Timestamp, parameters, paths]

=== EXFOR Data Loading ===
[Number of experiments loaded, sources used]

=== Energy Bin Processing ===
[For each energy bin:]
  E=X.XXXX MeV: N experiments, M points
    - Experiment details
    - Kernel diagnostics
    - Fit results

=== Interpolation Summary ===
[Bins with/without data, interpolation applied]

=== Experiments Used Summary ===
[Table of all experiments contributing to the evaluation]

=== MC Sampling ===
[Progress, timing, bin jump statistics if enabled]

=== Output Files Written ===
[List of generated files]
```

---

## 9. Troubleshooting

### 9.1 Common Issues and Solutions

#### "No EXFOR data" warnings for many energies

| Possible Cause | Solution |
|----------------|----------|
| Energy range outside data | Adjust `ENERGY_MIN_MEV`/`ENERGY_MAX_MEV` |
| Wrong ZAID | Check `TARGET_ZAIDS` includes correct isotope |
| Database path wrong | Verify `EXFOR_DB_PATH` or `EXFOR_DIRECTORY` |
| Data excluded | Review `EXCLUDE_EXPERIMENTS` list |
| Natural isotope not included | Add natural ZAID (e.g., 26000 for Fe) |

#### MF34 covariance is zero or very small

| Possible Cause | Solution |
|----------------|----------|
| Too few samples | Increase `N_SAMPLES` to at least 100 |
| Coefficients not varying | Check that data has real uncertainty |
| Wrong energy range | Verify coefficients are varying in log |

#### High chi-squared values ($\chi^2_{\text{red}} \gg 1$)

| Possible Cause | Solution |
|----------------|----------|
| Uncertainties underestimated | Increase `MIN_RELATIVE_UNCERTAINTY` |
| Discrepant experiments | Enable `USE_BAND_DISCREPANCY = True` |
| Single bad experiment | Add to `EXCLUDE_EXPERIMENTS` |
| Wrong reference frame | Check LAB/CM handling in log |

#### Oscillations in Legendre coefficients vs energy

| Possible Cause | Solution |
|----------------|----------|
| Over-fitting | Reduce `MAX_LEGENDRE_DEGREE` |
| Sparse data | Increase `GLOBAL_CONV_LAMBDA` |
| Experiment with many energies | Enable `DEDUPE_NOMINAL = True` |
| High-order instability | Increase `L_DEPENDENT_POWER` |

#### Energy jitter shows 0% bin jumps

| Possible Cause | Solution |
|----------------|----------|
| Energy grid too coarse | Normal for coarse grids |
| Resolution very small | Check TOF parameters in log |
| Jitter clipping too tight | Increase `JITTER_N_SIGMA_CLIP` |

### 9.2 Warning Messages Reference

| Warning | Meaning | Action |
|---------|---------|--------|
| `Using default TOF params for XXXXX` | No experiment-specific resolution | Add to `TOF_PARAMETERS_FILE` |
| `N_eff = X.X < threshold` | Few experiments dominate | Reduce `MAX_EXPERIMENT_WEIGHT_FRACTION` |
| `Weight sum = X.XX < threshold` | Data truncated at bin edge | Review data or reduce threshold |
| `Dataset skipped (weight_sum < 0.5)` | Severe truncation | Data mostly outside bin |
| `Interpolating from neighbors` | No EXFOR data at this energy | Consider adding more data sources |
| `Extrapolating outside data range` | Energy beyond EXFOR coverage | Expand data sources or narrow range |
| `Band τ capped at X%` | Large discrepancy limited | Review experiments in that band |

---

## 10. Technical Reference

### 10.1 Energy Resolution Formula

For TOF (Time-of-Flight) experiments:

$$
\sigma E = E \times 2 \times \frac{\Delta t}{t}
$$

where:
- $E$ = neutron energy (MeV)
- $\Delta t$ = time resolution (ns)
- $t = L / v$ = flight time (ns)
- $v = c \sqrt{2E/m_n}$ = neutron velocity (m/ns)
- $L$ = flight path (m)
- $m_n = 939.565$ MeV/c² = neutron mass
- $c = 0.2998$ m/ns = speed of light

In practical form:

$$
\sigma E \approx E \times \frac{2 \Delta t \cdot c \sqrt{2E/m_n}}{L}
$$

### 10.2 Angular Band Discrepancy Model

For each angular band $b \in \{F, M, B\}$:

1. **Compute normalized residuals:**
   $$r_i = \frac{y_i - y_{\text{fit},i}}{\sigma_i}$$

2. **Compute robust scale (MAD-based):**
   $$s_b = 1.4826 \times \text{median}(|r - \text{median}(r)|)$$

3. **Estimate band discrepancy:**
   $$\tau_b = \begin{cases}
   0 & \text{if } s_b \leq 1 \\
   \text{median}(\sigma_b) \times \sqrt{s_b^2 - 1} & \text{otherwise}
   \end{cases}$$

4. **Apply ceiling:**
   $$\tau_b \leftarrow \min(\tau_b, \text{MAX\_TAU\_FRACTION} \times \text{median}(|y_b|))$$

5. **Effective uncertainty:**
   $$\sigma_{\text{eff},i}^2 = \sigma_i^2 + \tau_b^2$$

### 10.3 Global Convolution System

The global system solves:

$$
\mathbf{c}^* = \arg\min_{\mathbf{c}} \left[ \|\mathbf{W}^{1/2}(\mathbf{A}\mathbf{c} - \mathbf{y})\|^2 + \lambda \|\mathbf{R}\mathbf{c}\|^2 \right]
$$

where:
- $\mathbf{c}$ = flattened coefficient vector $[c_0(E_1), c_1(E_1), \ldots, c_L(E_1), c_0(E_2), \ldots]$
- $\mathbf{A}$ = sparse design matrix (resolution-weighted Legendre values)
- $\mathbf{W}$ = diagonal weight matrix ($W_{ii} = 1/\sigma_i^2$)
- $\mathbf{y}$ = observation vector
- $\mathbf{R}$ = regularization matrix (second-difference with $\ell$-dependent scaling)

The regularization matrix $\mathbf{R}$ enforces:

$$
R_{k,\ell} = (1+\ell)^{p/2} \times [c_\ell(E_{k+1}) - 2c_\ell(E_k) + c_\ell(E_{k-1})]
$$

Solution via normal equations:

$$
(\mathbf{A}^T\mathbf{W}\mathbf{A} + \lambda \mathbf{R}^T\mathbf{R})\mathbf{c} = \mathbf{A}^T\mathbf{W}\mathbf{y}
$$

### 10.4 Per-Experiment Normalization

In MC sampling, experiment normalization factors are drawn:

**Lognormal (recommended):**
$$
f_k \sim \text{LogNormal}(-\frac{\sigma_n^2}{2}, \sigma_n)
$$

where $\sigma_n$ = `NORMALIZATION_SIGMA`. The mean is 1.0 by construction.

**Normal:**
$$
f_k \sim \mathcal{N}(1, \sigma_n)
$$

The cross section is then scaled: $y'_i = f_k \times y_i$ for all points from experiment $k$.

### 10.5 Overlap Weight Calculation

For resolution-aware weighting:

$$
w_{jk} = \Phi\left(\frac{E_{k,\text{high}} - E_j}{\sigma_j}\right) - \Phi\left(\frac{E_{k,\text{low}} - E_j}{\sigma_j}\right)
$$

where:
- $\Phi$ = standard normal CDF
- $E_j$ = EXFOR measurement energy
- $[E_{k,\text{low}}, E_{k,\text{high}}]$ = ENDF bin boundaries
- $\sigma_j$ = experiment-specific energy resolution

Properties:
- $w_{jk} \in [0, 1]$
- $\sum_k w_{jk} \leq 1$ (< 1 if data truncated at range edges)
- $w_{jk} \approx 1$ when measurement well inside bin
- $w_{jk} \approx 0$ when measurement clearly outside bin

---

## References

- EXFOR database: https://www-nds.iaea.org/exfor/
- ENDF format manual: https://www.nndc.bnl.gov/csewg/docs/endf-manual.pdf
- kika.exfor module documentation: See `kika/exfor/README.md`
