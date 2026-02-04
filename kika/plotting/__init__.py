"""
Plotting infrastructure for KIKA.

This module provides a flexible, object-oriented approach to creating plots
by separating data representation from visual styling and plot composition.
"""

from .plot_data import (
    PlotData,
    LegendreCoeffPlotData,
    LegendreUncertaintyPlotData,
    AngularDistributionPlotData,
    CrossSectionPlotData,
    MultigroupXSPlotData,
    MultigroupUncertaintyPlotData,
    UncertaintyBand,
    HeatmapPlotData,
    CovarianceHeatmapData,
    MF34HeatmapData,
)
from .plot_builder import PlotBuilder
from .heatmap_builder import HeatmapBuilder

__all__ = [
    'PlotData',
    'LegendreCoeffPlotData',
    'LegendreUncertaintyPlotData',
    'AngularDistributionPlotData',
    'CrossSectionPlotData',
    'MultigroupXSPlotData',
    'MultigroupUncertaintyPlotData',
    'UncertaintyBand',
    'HeatmapPlotData',
    'CovarianceHeatmapData',
    'MF34HeatmapData',
    'PlotBuilder',
    'HeatmapBuilder',
]
