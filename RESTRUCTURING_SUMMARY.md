# KIKA Covariance and Plotting Restructuring - Complete

**Date:** 2024-2025
**Status:** ✅ COMPLETE AND VERIFIED (Updated: December 11, 2025)

## Overview

Successfully restructured the `kika.cov` and `kika.plotting` directories following best practices:
- Modern PlotBuilder-based implementations
- Legacy code preservation
- Full backward compatibility
- Clean separation of concerns

## Recent Updates (December 2025)

### Unified Uncertainty Handling API for PlotBuilder and ENDF

**Problem:** The plotting API had inconsistent uncertainty handling across ENDF classes (MF4 vs MF34), with boolean uncertainty parameters causing confusion and blank plots during visualization.

**Solution:** Implemented comprehensive API redesign with unified sigma-based uncertainty control:

#### 1. ENDF.to_plot_data() Refactoring
- **Location:** [endf.py](kika/endf/classes/endf.py#L67)
- **Change:** Now **always returns tuple** `(PlotData, UncertaintyPlotData | None)` for MF4
- **Removed:** Boolean `uncertainty` parameter (avoided confusion about which data to return)
- **Added:** `sigma` parameter (default=1.0) for controlling uncertainty scaling
- **Behavior:** Automatically fetches MF34 data when available; returns uncertainty as second element of tuple
- **For MF34:** Returns uncertainty data only (single object, not tuple)

#### 2. PlotBuilder.add_data() Enhancement
- **Location:** [plot_builder.py](kika/plotting/plot_builder.py#L195)
- **New Parameter:** `sigma` parameter with three modes:
  - `sigma=None` (default): Use sigma from uncertainty data source
  - `sigma=0`: Skip plotting uncertainty bands (even if provided)
  - `sigma=2.0`: Scale uncertainties by specified sigma level
- **Tuple Support:** Directly accepts tuples from `to_plot_data()` method
- **Fix:** Properly filters sigma from matplotlib styling arguments (line 261)

#### 3. LegendreUncertaintyPlotData Enhancement
- **Location:** [plot_data.py](kika/plotting/plot_data.py)
- **Added:** `sigma` attribute (default=1.0) for tracking uncertainty scaling level
- **Purpose:** Stores sigma level metadata for uncertainty visualization

#### 4. MF34CovMat.to_plot_data() Update
- **Location:** [mf34_covmat.py](kika/cov/mf34_covmat.py)
- **Added:** `sigma` parameter for consistent API across covariance classes
- **Behavior:** Applies sigma scaling when returning uncertainty data

#### 5. MF4 Polynomial/Mixed Classes Fixes
- **Locations:** [polynomial.py](kika/endf/classes/mf4/polynomial.py), [mixed.py](kika/endf/classes/mf4/mixed.py)
- **Problem:** Both had broken imports from non-existent `.plot_utils` module
- **Solution:** Now use direct `LegendreCoeffPlotData` implementation
- **Result:** Both return proper PlotData objects compatible with PlotBuilder

**Testing Results:**
- ✅ All three plot types working correctly: nominal only, nominal+uncertainties, uncertainties-only
- ✅ Sigma parameter correctly controls uncertainty band visualization
- ✅ Uncertainty data from MF34 properly integrated with MF4 nominal data
- ✅ Backward compatibility maintained for existing code using old signatures

**Example Usage:**
```python
from kika.plotting import PlotBuilder

# Get data WITH uncertainties - always returns tuple
data, unc = endf.to_plot_data(mf=4, mt=2, order=1)

# Plot nominal data without uncertainty bands
fig = (PlotBuilder()
    .add_data(data, uncertainty=unc, sigma=0)
    .set_labels(title='Cross Section')
    .build())

# Plot nominal data WITH 1-sigma uncertainty bands (default)
fig = (PlotBuilder()
    .add_data(data, uncertainty=unc)
    .set_labels(title='Cross Section with 1σ Uncertainties')
    .build())

# Plot ONLY uncertainty values as lines (from MF34)
unc_only = endf.to_plot_data(mf=34, mt=2, order=1)
fig = (PlotBuilder()
    .add_data(unc_only)
    .set_labels(title='Angular Distribution Uncertainties')
    .build())

# Multiple data sets with custom sigma
fig = (PlotBuilder()
    .add_data(data1, uncertainty=unc1, sigma=1.0, label='JEFF-4.0')
    .add_data(data2, uncertainty=unc2, sigma=2.0, label='JENDL-5 (2σ)')
    .add_data(data3, uncertainty=unc3, sigma=0, label='TENDL-2023 (no band)')
    .build())
```

## What Was Changed

### 1. Modern Plotting Implementations Created

#### `kika.plotting.covariance` (4 functions)
- `plot_covariance_heatmap()` - Plots covariance matrix heatmaps
- `plot_mf34_covariance_heatmap()` - Plots MF34 covariance heatmaps
- `plot_uncertainties()` - Plots uncertainty profiles for multiple (zaid, mt) pairs
- `plot_multigroup_xs()` - Plots multigroup cross sections with optional uncertainties

#### `kika.plotting.multigroup_covariance` (4 functions)
- `plot_mg_legendre_coefficients()` - Plots multigroup Legendre coefficients with uncertainties
- `plot_mg_vs_endf_comparison()` - Compares MG data with ENDF continuous energy data
- `plot_mg_vs_endf_uncertainties_comparison()` - Compares uncertainty profiles (MG vs ENDF)
- `plot_mg_covariance_heatmap()` - Plots multigroup covariance heatmaps

All functions use the PlotBuilder pattern and integrate with data class `to_plot_data()` and `to_heatmap_data()` methods.

### 2. Legacy Code Relocation

Moved to `kika.cov.legacy/`:
- `legacy_covmat_plotting.py` (1351 lines) - Heatmap functions
- `legacy_plotting.py` (913 lines) - Uncertainty and multigroup XS plotting
- `mf34_plotting.py` - MF34 heatmap functions

Moved to `kika.cov.multigroup/`:
- `legacy_mg_plotting.py` (1551 lines) - All multigroup plotting implementations

### 3. Deprecation Shims for Backward Compatibility

Created shim files that import from legacy locations with deprecation warnings:
- `kika.cov.heatmap` → imports from `legacy.legacy_covmat_plotting`
- `kika.cov.plotting` → imports from `legacy.legacy_plotting`
- `kika.cov.mf34cov_heatmap` → imports from `legacy.mf34_plotting`
- `kika.cov.multigroup.plotting_mg` → imports from `legacy_mg_plotting`

### 4. Data Class Method Updates

#### `CovMat` class (`covmat.py`)
- `plot_uncertainties()` → delegates to `kika.plotting.covariance.plot_uncertainties`
- `plot_multigroup_xs()` → delegates to `kika.plotting.covariance.plot_multigroup_xs`
- `plot_covariance_heatmap()` → already delegated to modern implementation

#### `MGMF34CovMat` class (`multigroup/mg_mf34_covmat.py`)
- `plot_legendre_coefficients()` → delegates to `kika.plotting.multigroup_covariance.plot_mg_legendre_coefficients`
- `plot_vs_endf()` → delegates to `kika.plotting.multigroup_covariance.plot_mg_vs_endf_comparison`
- `plot_covariance_heatmap()` → delegates to `kika.plotting.multigroup_covariance.plot_mg_covariance_heatmap`
- `plot_uncertainties_comparison()` → delegates to `kika.plotting.multigroup_covariance.plot_mg_vs_endf_uncertainties_comparison`

All methods now use modern PlotBuilder-based implementations while maintaining their original APIs.

### 5. Import Fixes

Fixed broken import in `legacy_mg_plotting.py`:
- Changed `from ..._plot_settings` to `from kika.plotting.styles`

### 6. Module Exports

Updated `kika.plotting.__init__.py` to export:
- `CovarianceHeatmapData`
- `MF34HeatmapData`
- `HeatmapPlotData`
- All PlotBuilder infrastructure

## Testing Results

All 6 comprehensive tests passed:
- ✅ Modern covariance functions
- ✅ Modern multigroup functions
- ✅ Data class plotting methods
- ✅ Legacy imports via shims
- ✅ PlotBuilder infrastructure
- ✅ Method delegation verification

## Usage Examples

### Modern API (Recommended)

```python
from kika.plotting.covariance import plot_uncertainties, plot_multigroup_xs
from kika.plotting.multigroup_covariance import plot_mg_legendre_coefficients
from kika.plotting import PlotBuilder

# Use standalone functions
fig = plot_uncertainties(covmat, zaid_mt_pairs=[(1001, 1), (1001, 2)])
fig = plot_mg_legendre_coefficients(mg_covmat, isotope=1001, mt=2, orders=[0, 1, 2])

# Or use data class methods (same modern implementation)
fig = covmat.plot_uncertainties(zaid_mt_pairs=[(1001, 1)])
fig = mg_covmat.plot_legendre_coefficients(isotope=1001, mt=2)
```

### Legacy API (Still Supported)

```python
# Old imports still work via deprecation shims
from kika.cov.plotting import plot_uncertainties  # Shows DeprecationWarning
from kika.cov.multigroup.plotting_mg import plot_mg_legendre_coefficients

# Or via multigroup module
from kika.cov.multigroup import plot_mg_legendre_coefficients
```

## Key Design Patterns

1. **PlotBuilder Pattern**: All modern functions use PlotBuilder for consistent plot creation
2. **Data Converter Methods**: Data classes provide `to_plot_data()` and `to_heatmap_data()` methods
3. **Delegation**: Class methods delegate to standalone functions for cleaner code
4. **Backward Compatibility**: Shims ensure old code continues to work with deprecation warnings

## Files Modified

### Created
- `kika/plotting/covariance.py` (extended)
- `kika/plotting/multigroup_covariance.py` (new)
- `kika/cov/heatmap.py` (shim)
- `kika/cov/plotting.py` (shim)
- `kika/cov/mf34cov_heatmap.py` (shim)
- `kika/cov/multigroup/plotting_mg.py` (shim)

### Relocated
- `kika/cov/legacy/legacy_covmat_plotting.py`
- `kika/cov/legacy/legacy_plotting.py`
- `kika/cov/legacy/mf34_plotting.py`
- `kika/cov/multigroup/legacy_mg_plotting.py`

### Modified
- `kika/cov/covmat.py` (updated 2 methods)
- `kika/cov/multigroup/mg_mf34_covmat.py` (updated 4 methods)
- `kika/cov/legacy/__init__.py` (added exports)
- `kika/plotting/__init__.py` (added exports)

## Migration Guide for Users

### For New Code
Use the modern API from `kika.plotting.covariance` and `kika.plotting.multigroup_covariance`:

```python
from kika.plotting.covariance import plot_uncertainties
from kika.plotting.multigroup_covariance import plot_mg_legendre_coefficients
```

### For Existing Code
No changes needed! Old imports will continue to work, but you'll see deprecation warnings encouraging migration to the modern API.

### To Update Existing Code
Simply change imports:
```python
# Old
from kika.cov.plotting import plot_uncertainties

# New
from kika.plotting.covariance import plot_uncertainties
```

Or use class methods directly (they now use modern implementations):
```python
fig = covmat.plot_uncertainties(...)
```

## Benefits

1. **Cleaner Architecture**: PlotBuilder pattern provides consistent plotting interface
2. **Better Maintainability**: Modern code is more readable and testable
3. **Full Compatibility**: All existing code continues to work
4. **Future-Proof**: Easy to extend with new plot types
5. **Type Safety**: Modern implementations have proper type hints
6. **Consistent Returns**: All functions return `plt.Figure` objects

## Notes

- All deprecation warnings can be silenced if needed
- Legacy code is preserved and fully functional
- No breaking changes for end users
- Modern implementations leverage existing `to_plot_data()` infrastructure
- All tests pass with zero errors in modified files

## Next Steps - Recommended Improvements for Plotting Infrastructure

### 1. Extend Unified API to All MF Sections
**Status:** Partial ✓ (Currently: MF4, MF34)
**Scope:** Apply the sigma-based uncertainty pattern to other MF sections
- **MF1:** Decay data with uncertainties
- **MF3:** Cross section with uncertainties  
- **MF12/14:** Photon data with uncertainties
- **MF15:** Energy distribution with uncertainties

**Impact:** Consistent plotting API across all data types  
**Effort:** Medium (pattern is established, apply to remaining classes)

### 2. Enhance UncertaintyBand with More Visualization Modes
**Current:** Relative and absolute uncertainties with alpha blending

**Proposed:**
- Covariance-based visualization (heatmaps + error ellipses)
- Percentile bands (5%, 16%, 50%, 84%, 95%)
- Bayesian posterior distributions
- Multi-model ensemble uncertainty visualization

**Implementation:** Extend `UncertaintyBand` class with additional plotting modes  
**Effort:** Medium-High (requires new visualization logic)

### 3. Add Interactive Features via Plotly/Bokeh Backend
**Proposed Enhancements:**
- Interactive sigma slider (1σ to 3σ real-time adjustment)
- Hover tooltips showing exact values and uncertainties
- Zoom/pan with uncertainty band updates
- Data point identification and annotation tools
- Toggle individual datasets on/off

**Technology:** Integrate with Plotly/Bokeh as alternative backends to matplotlib  
**Effort:** High (requires interactive backend integration)

### 4. Expand Legend and Labeling System
**Current:** Basic legend with single label per dataset

**Proposed:**
- Auto-generated labels with isotope, reaction type, and data source
- Uncertainty notation in legend (e.g., "JEFF-4.0 (±5%)")
- Configurable label formats (isotope, ZAID, reaction name, sigma info)
- Multi-line legends with metadata columns (library, evaluation, version)

**Implementation:** Enhanced legend factory methods in PlotBuilder  
**Effort:** Low-Medium

### 5. Add Comparison Plots with Relative Differences
**Functionality:**
- Plot (data1 - data2) / data2 relative differences
- Unified σ-based difference bands
- Highlight agreement regions (< 1σ difference)
- Separate subplot showing absolute vs relative differences

**Use Cases:** Evaluate data library differences (JEFF vs JENDL vs TENDL)  
**Implementation:** New specialized plot types in `kika.plotting.covariance`  
**Effort:** Low

### 6. Implement Export Formats
**Current:** Matplotlib figure only

**Proposed:**
- PDF/EPS with embedded uncertainties (bands + transparency)
- HDF5 format with all plot data + metadata + uncertainties
- JSON with complete plot configuration for reproducibility
- CSV with plotted values for external analysis tools

**Implementation:** Export methods on PlotBuilder class  
**Effort:** Low-Medium

### 7. Optimize Performance for Large Datasets
**Scenario:** Plotting 1000+ energy points × 5 libraries × uncertainty bands

**Improvements:**
- Lazy loading of uncertainty data (only when visible)
- Intelligent decimation/sampling for high-resolution plots
- GPU-accelerated uncertainty band rendering
- Progressive rendering (nominal first, then uncertainties)

**Implementation:** Add performance mode to PlotBuilder constructor  
**Effort:** Medium-High

### 8. Add Validation and Quality Checks
**Proposed Features:**
- Warn when sigma > 3 (physically unrealistic)
- Check for negative uncertainties
- Verify uncertainty data aligns with nominal data points
- Auto-flag inconsistent sigmas across datasets
- Validation report before plotting

**Implementation:** Validation methods called before rendering  
**Effort:** Low

### 9. Standardize Uncertainty Attribution and Tracking
**Current:** Sigma stored in individual data objects

**Proposed:**
- Sigma metadata tracking (source: MF34, model, parametric)
- Multiple uncertainty sources per dataset
- Uncertainty budget breakdown visualization
- Correlation information display
- Traceable uncertainty lineage

**Use Cases:** Quality assurance, uncertainty decomposition, validation  
**Effort:** Medium

### 10. Create Advanced Examples and Tutorials
**Priority:** High (improves user adoption and understanding)

**Content:**
- Example: Plotting with different sigma levels
- Example: Uncertainty-only plots for QA workflows
- Example: Multi-library comparison plots
- Example: Relative difference plots (JEFF vs JENDL)
- Tutorial: Creating publication-quality plots
- Tutorial: Custom uncertainty visualizations
- Tutorial: Batch plotting scripts

**Implementation:** Jupyter notebooks in `examples/` directory  
**Effort:** Low-Medium

## Implementation Priority Roadmap

### Immediate (1-2 weeks)
1. Extend unified API to MF3, MF1 (most commonly used sections)
2. Create advanced examples and tutorials
3. Implement validation checks for uncertainties

### Short-term (1-2 months)
4. Add comparison plots with relative differences
5. Enhance legend/labeling system
6. Implement basic export formats (CSV, PDF, JSON)

### Medium-term (2-4 months)
7. Add interactive features (sigma slider, tooltips)
8. Enhance UncertaintyBand visualization modes
9. Performance optimization for large datasets

### Long-term (4+ months)
10. GPU acceleration for batch processing
11. Advanced statistical visualizations
12. Full interactive dashboard interface
