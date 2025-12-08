# KIKA Covariance and Plotting Restructuring - Complete

**Date:** 2024
**Status:** ✅ COMPLETE AND VERIFIED

## Overview

Successfully restructured the `kika.cov` and `kika.plotting` directories following best practices:
- Modern PlotBuilder-based implementations
- Legacy code preservation
- Full backward compatibility
- Clean separation of concerns

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
