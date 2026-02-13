"""
Deprecated shim for backward compatibility.

This module has been moved to kika.cov.multigroup.legacy_mg_plotting.
For new code, use kika.plotting with PlotBuilder instead.
"""

import warnings

warnings.warn(
    "kika.cov.multigroup.plotting_mg is deprecated and will be removed in a future version. "
    "Use modern plotting APIs with PlotBuilder instead.",
    DeprecationWarning,
    stacklevel=2
)

from kika.cov.multigroup.legacy_mg_plotting import (
    plot_mg_legendre_coefficients,
    plot_mg_vs_endf_comparison,
    plot_mg_vs_endf_uncertainties_comparison,
    plot_mg_covariance_heatmap,
)

__all__ = [
    'plot_mg_legendre_coefficients',
    'plot_mg_vs_endf_comparison',
    'plot_mg_vs_endf_uncertainties_comparison',
    'plot_mg_covariance_heatmap',
]
