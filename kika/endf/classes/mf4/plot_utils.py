"""
Plotting utilities for MF4 (Angular Distributions) data.

This module provides helper functions for creating plottable data objects
from MF4 sections (Legendre coefficients and mixed representations).
"""

from typing import Optional, Tuple
import numpy as np


def create_legendre_coeff_plot_data(
    mf4_section,
    order: int,
    label: Optional[str] = None,
    **styling_kwargs
):
    """
    Create a LegendreCoeffPlotData object from an MF4 section.

    This function extracts energy-dependent Legendre coefficients from
    MF4MTLegendre or MF4MTMixed objects and creates a plottable data object.

    Parameters
    ----------
    mf4_section : MF4MTLegendre or MF4MTMixed
        The MF4 section containing Legendre coefficient data.
        Must have `_energies` and `_legendre_coeffs` attributes.
    order : int
        Legendre polynomial order to extract (0, 1, 2, ...).
        Order 0 returns 1.0 for all energies (ENDF convention: a_0 = 1).
        Higher orders are extracted from stored coefficients.
    label : str, optional
        Custom label for the plot. If None, auto-generates from isotope and order.
    **styling_kwargs
        Additional styling kwargs (color, linestyle, linewidth, marker, etc.)

    Returns
    -------
    LegendreCoeffPlotData
        Plot data object ready to be added to a PlotBuilder

    Examples
    --------
    >>> # From MF4MTLegendre
    >>> data = create_legendre_coeff_plot_data(mf4_legendre, order=1, color='blue')
    >>> builder = PlotBuilder().add_data(data).build()
    >>>
    >>> # From MF4MTMixed (uses Legendre portion)
    >>> data = create_legendre_coeff_plot_data(mf4_mixed, order=2, linestyle='--')

    Notes
    -----
    ENDF Convention:
    - a_0 = 1 is implicit (normalized probability distribution)
    - Stored coefficients are a_1, a_2, ... at each energy point
    - Missing higher-order coefficients are treated as zero

    For MF4MTMixed (LTT=3), only the Legendre portion is used.
    The tabulated portion at higher energies is not included.
    """
    from kika.plotting import LegendreCoeffPlotData

    # Extract energy grid
    energies = np.asarray(mf4_section._energies, dtype=float)

    if energies.size == 0:
        raise ValueError("No energy data available in MF4 section")

    # Extract coefficients for the specified order
    coeff_lists = mf4_section._legendre_coeffs
    num_energies = len(energies)
    coeffs = np.zeros(num_energies, dtype=float)

    if order == 0:
        # a_0 = 1.0 by ENDF convention (normalized PDF)
        coeffs = np.ones(num_energies, dtype=float)
    else:
        # For order L >= 1, coefficient is at index L-1 in stored list
        coeff_index = order - 1
        for i, coeff_list in enumerate(coeff_lists):
            if coeff_list and coeff_index < len(coeff_list):
                coeffs[i] = coeff_list[coeff_index]
            # else: remains 0.0 (missing coefficients are zero)

    # Get isotope information
    isotope = getattr(mf4_section, 'isotope', None)
    if isotope is None and hasattr(mf4_section, 'zaid'):
        isotope = str(mf4_section.zaid)

    # Get MT number
    mt = getattr(mf4_section, 'number', None)

    # Determine energy range
    energy_range = (float(energies.min()), float(energies.max()))

    # Auto-generate label if not provided
    if label is None:
        from kika._constants import format_plot_label
        label = format_plot_label(isotope=isotope, mt=mt, order=order)

    return LegendreCoeffPlotData(
        x=energies,
        y=coeffs,
        order=order,
        isotope=isotope,
        mt=mt,
        energy_range=energy_range,
        label=label,
        **styling_kwargs
    )
