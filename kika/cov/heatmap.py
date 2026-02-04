"""
Deprecated shim for backward compatibility.

This module has been moved to kika.cov.legacy.legacy_covmat_plotting.
For new code, use kika.plotting.covariance.plot_covariance_heatmap instead.
"""

import warnings

warnings.warn(
    "kika.cov.heatmap is deprecated and will be removed in a future version. "
    "Use kika.plotting.covariance instead for modern plotting functionality.",
    DeprecationWarning,
    stacklevel=2
)

from kika.cov.legacy.legacy_covmat_plotting import (
    plot_covariance_heatmap,
    plot_covariance_difference_heatmap,
    _setup_energy_group_ticks,
    _setup_energy_group_ticks_single_block,
)

__all__ = [
    'plot_covariance_heatmap',
    'plot_covariance_difference_heatmap',
]
