"""
Deprecated shim for backward compatibility.

This module has been moved to kika.cov.legacy.legacy_plotting.
For new code, use kika.plotting.covariance and PlotBuilder instead.
"""

import warnings

warnings.warn(
    "kika.cov.plotting is deprecated and will be removed in a future version. "
    "Use kika.plotting.covariance and PlotBuilder for modern plotting functionality.",
    DeprecationWarning,
    stacklevel=2
)

from kika.cov.legacy.legacy_plotting import (
    plot_uncertainties,
    plot_multigroup_xs,
)

__all__ = [
    'plot_uncertainties',
    'plot_multigroup_xs',
]
