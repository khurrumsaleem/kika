"""
Covariance matrix plotting functions using the modern PlotBuilder infrastructure.

This module provides convenience wrapper functions for plotting covariance heatmaps
using the refactored plotting system with to_heatmap_data() methods and PlotBuilder.

The functions maintain backward compatibility with the original API while using
the new, cleaner implementation.
"""

import matplotlib.pyplot as plt
from typing import Union, Sequence, Tuple, List, Optional
from kika.cov.covmat import CovMat
from kika.cov.mf34_covmat import MF34CovMat
from kika.plotting.heatmap_builder import HeatmapBuilder
from kika.plotting.plot_builder import PlotBuilder


def plot_covariance_heatmap(
    covmat: CovMat,
    nuclide: Union[int, str, List[Union[int, str]]],
    mt: Union[int, Sequence[int], Tuple[int, int]],
    *,
    matrix_type: str = "corr",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = True,
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Draw a covariance or correlation matrix heatmap for one or more isotopes and
    one or more MT reactions, with optional uncertainty plots shown above the heatmap columns.

    This function uses the modern PlotBuilder infrastructure for cleaner, more
    maintainable code while maintaining the same API as the original implementation.

    Parameters
    ----------
    covmat : CovMat
        The covariance matrix object
    nuclide : int, str, or list of int/str
        Isotope identifier(s). Can be:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
        - List of ZAIDs or strings for multi-isotope heatmaps (e.g., ['Fe54', 'Fe56'])
    mt : int, sequence of int, or tuple of (row_mt, col_mt)
        MT reaction number(s). Can be:
        - Single int: diagonal block for that MT
        - Sequence of ints: diagonal blocks for those MTs
        - Tuple of (row_mt, col_mt): off-diagonal block between row and column MT
          (not supported for multi-isotope heatmaps)
    matrix_type : str, default "corr"
        Type of matrix to plot: "corr"/"correlation" for correlation matrix,
        or "cov"/"covariance" for covariance matrix
    figsize : tuple, default (6, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    vmax, vmin : float, optional
        DEPRECATED: These parameters are ignored. Auto-scaling is used.
    show_uncertainties : bool, default True
        Whether to show uncertainty plots above the heatmap
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear'
    energy_range : tuple of float, optional
        Energy range (min, max) for filtering. Values in eV.

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots

    Raises
    ------
    ValueError
        If no data found for specified isotope/MT combination

    Examples
    --------
    Plot a single MT diagonal block:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=92235, mt=2)
    >>> plt.savefig("u235_elastic.png")

    Plot with nuclide string:

    >>> fig = plot_covariance_heatmap(covmat, nuclide='U235', mt=[2, 18, 102])

    Plot covariance matrix instead of correlation:

    >>> fig = plot_covariance_heatmap(covmat, nuclide='Fe56', mt=2, matrix_type='cov')

    Plot an off-diagonal block between two MTs:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=92235, mt=(2, 18))

    Plot multi-isotope heatmap showing cross-isotope correlations:

    >>> fig = plot_covariance_heatmap(covmat, nuclide=['Fe54', 'Fe56'], mt=[2, 18])

    See Also
    --------
    plot_covariance_difference_heatmap : Plot differences between two covariance matrices
    plot_mf34_covariance_heatmap : Plot MF34 angular distribution covariances
    """
    # Note: nuclide conversion and multi-isotope handling is done in to_heatmap_data()

    # Normalize matrix_type
    matrix_type_normalized = matrix_type.lower()
    if matrix_type_normalized in ("corr", "correlation"):
        matrix_type_normalized = "corr"
    elif matrix_type_normalized in ("cov", "covariance"):
        matrix_type_normalized = "cov"
    else:
        raise ValueError(f"matrix_type must be 'corr'/'correlation' or 'cov'/'covariance', got '{matrix_type}'")
    
    # Normalize scale parameter
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        scale_normalized = "log"
    elif scale_normalized in ("lin", "linear"):
        scale_normalized = "linear"
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")
    
    # Prepare the heatmap data using the new infrastructure
    heatmap_data = covmat.to_heatmap_data(
        nuclide=nuclide,
        mt=mt,
        matrix_type=matrix_type_normalized,
        scale=scale_normalized,
        energy_range=energy_range,
    )
    
    # Warn about deprecated vmin/vmax parameters
    import warnings
    if vmin is not None or vmax is not None:
        warnings.warn(
            "vmin and vmax parameters are deprecated and will be ignored. "
            "Auto-scaling is now used for colorbar normalization.",
            DeprecationWarning,
            stacklevel=2
        )

    # Create the plot using HeatmapBuilder (always use light style for heatmaps)
    builder = HeatmapBuilder(style="light", figsize=figsize, dpi=dpi, font_family=font_family)
    fig = builder.add_heatmap(
        heatmap_data,
        show_uncertainties=show_uncertainties,
    ).build()
    # If uncertainties panels are shown, the HeatmapBuilder places the suptitle
    # a bit high; nudge it down slightly for better layout.
    if show_uncertainties and fig is not None:
        # Determine effective title used by builder/heatmap_data
        effective_title = getattr(builder, "_title", None) or getattr(heatmap_data, "label", None)
        if effective_title:
            try:
                fig.suptitle(effective_title, y=0.94)
            except Exception:
                # Fallback: ignore if suptitle fails for any reason
                pass

    return fig


def plot_mf34_covariance_heatmap(
    mf34_covmat: MF34CovMat,
    nuclide: Union[int, str],
    mt: int,
    legendre_coeffs: Union[int, List[int], Tuple[int, int]],
    *,
    matrix_type: str = "corr",
    figsize: Tuple[float, float] = (6, 6),
    dpi: int = 300,
    font_family: str = "serif",
    vmax: float | None = None,
    vmin: float | None = None,
    show_uncertainties: bool = False,
    cmap: Optional[str] = None,
    scale: str = "log",
    energy_range: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Draw a covariance/correlation heatmap for MF34 angular distribution data
    with energy-proportional blocks and optional uncertainty panels.

    This function handles the more complex MF34 covariance structure where each
    Legendre coefficient can have a different energy grid.

    Parameters
    ----------
    mf34_covmat : MF34CovMat
        The MF34 covariance matrix object
    nuclide : int or str
        Isotope identifier. Can be either:
        - Integer ZAID (e.g., 92235 for U-235)
        - Element-mass string (e.g., 'U235', 'Fe56')
    mt : int
        Reaction MT number
    legendre_coeffs : int, list of int, or tuple of (L1, L2)
        Legendre coefficient(s) to plot. Can be:
        - Single int: diagonal block for that L
        - List of ints: diagonal blocks for those L values
        - Tuple of (L1, L2): off-diagonal block between L1 and L2
    matrix_type : str, default "corr"
        Matrix type to plot: "corr"/"correlation" for correlation matrix,
        or "cov"/"covariance" for covariance matrix
    figsize : tuple, default (6, 6)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    vmax, vmin : float, optional
        DEPRECATED: These parameters are ignored. Auto-scaling is used.
    show_uncertainties : bool, default False
        Whether to show uncertainty plots above the heatmap
    cmap : str, optional
        Colormap name (e.g., 'viridis', 'RdYlGn')
    scale : str, default "log"
        Energy axis scale: "log"/"logarithmic" or "lin"/"linear"
    energy_range : tuple of float, optional
        Energy range (min, max) for filtering. Values in eV.

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the heatmap and optional uncertainty plots

    Raises
    ------
    ValueError
        If no data found for specified isotope/MT/Legendre combination

    Examples
    --------
    Plot correlation matrix for Legendre coefficients L=1,2,3:
    
    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide=92235, mt=2,
    ...     legendre_coeffs=[1, 2, 3]
    ... )

    Plot with nuclide string:
    
    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide='U235', mt=2,
    ...     legendre_coeffs=[1, 2, 3]
    ... )

    Plot covariance for a single Legendre coefficient with uncertainties:
    
    >>> fig = plot_mf34_covariance_heatmap(
    ...     mf34_covmat, nuclide='Fe56', mt=2,
    ...     legendre_coeffs=1, matrix_type="cov",
    ...     show_uncertainties=True
    ... )

    See Also
    --------
    plot_covariance_heatmap : Plot multigroup cross-section covariances
    """
    # Convert nuclide to ZAID if string
    from kika._utils import symbol_to_zaid
    
    if isinstance(nuclide, str):
        isotope = symbol_to_zaid(nuclide)
    else:
        isotope = nuclide
    
    # Normalize matrix_type
    matrix_type_normalized = matrix_type.lower()
    if matrix_type_normalized in ("corr", "correlation"):
        matrix_type_normalized = "corr"
    elif matrix_type_normalized in ("cov", "covariance"):
        matrix_type_normalized = "cov"
    else:
        raise ValueError(f"matrix_type must be 'corr'/'correlation' or 'cov'/'covariance', got '{matrix_type}'")
    
    # Normalize scale
    scale_normalized = scale.lower()
    if scale_normalized in ("log", "logarithmic"):
        scale_normalized = "log"
    elif scale_normalized in ("lin", "linear"):
        scale_normalized = "linear"
    else:
        raise ValueError(f"scale must be 'log'/'logarithmic' or 'lin'/'linear', got '{scale}'")
    
    # Prepare the heatmap data using the new infrastructure
    heatmap_data = mf34_covmat.to_heatmap_data(
        nuclide=nuclide,
        mt=mt,
        legendre_coeffs=legendre_coeffs,
        matrix_type=matrix_type_normalized,
        scale=scale_normalized,
        energy_range=energy_range,
    )
    
    # Override colormap if provided
    if cmap is not None:
        heatmap_data.cmap = cmap

    # Warn about deprecated vmin/vmax parameters
    import warnings
    if vmin is not None or vmax is not None:
        warnings.warn(
            "vmin and vmax parameters are deprecated and will be ignored. "
            "Auto-scaling is now used for colorbar normalization.",
            DeprecationWarning,
            stacklevel=2
        )

    # Create the plot using HeatmapBuilder (always use light style for heatmaps)
    builder = HeatmapBuilder(style="light", figsize=figsize, dpi=dpi, font_family=font_family)
    fig = builder.add_heatmap(
        heatmap_data,
        show_uncertainties=show_uncertainties,
    ).build()

    # If uncertainties panels are shown, the HeatmapBuilder places the suptitle
    # a bit high; nudge it down slightly for better layout.
    if show_uncertainties and fig is not None:
        # Determine effective title used by builder/heatmap_data
        effective_title = getattr(builder, "_title", None) or getattr(heatmap_data, "label", None)
        if effective_title:
            try:
                fig.suptitle(effective_title, y=0.93)
            except Exception:
                # Fallback: ignore if suptitle fails for any reason
                pass
    
    return fig


def plot_uncertainties(
    covmat: CovMat,
    nuclide: Union[int, str, Sequence[Union[int, str]]],
    mt: Union[int, Sequence[int]],
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    sigma: float = 1.0,
    style: str = "light",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    xscale: str = "log",
    yscale: str = "linear",
    title: Optional[str] = "default",
    **styling_kwargs
) -> plt.Figure:
    """
    Plot relative uncertainties for one or more (ZAID, MT) pairs from covariance data.
    
    This modern implementation uses the PlotBuilder infrastructure with to_plot_data()
    for cleaner, more maintainable code.

    Parameters
    ----------
    covmat : CovMat
        The covariance matrix object
    zaid : int or sequence of int
        Isotope ID(s) to plot (e.g., 92235 for U-235)
    mt : int or sequence of int
        Reaction MT number(s) to plot
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis in MeV. If None, uses full range.
    sigma : float, default 1.0
        Number of sigma levels for uncertainty (e.g., 1.0 for 1σ, 2.0 for 2σ)
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    **styling_kwargs
        Additional styling arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the uncertainty plots

    Examples
    --------
    Plot uncertainties for a single reaction:
    
    >>> fig = plot_uncertainties(covmat, zaid=92235, mt=2)
    
    Plot multiple reactions:
    
    >>> fig = plot_uncertainties(covmat, zaid=92235, mt=[2, 18, 102])
    
    Plot with custom styling:
    
    >>> fig = plot_uncertainties(
    ...     covmat, zaid=92235, mt=2,
    ...     sigma=2.0, style='presentation'
    ... )

    See Also
    --------
    plot_multigroup_xs : Plot cross sections with optional uncertainties
    plot_covariance_heatmap : Plot covariance heatmaps
    """
    # Normalize inputs to lists and convert nuclide symbols to ZAID
    from kika._utils import symbol_to_zaid, zaid_to_symbol

    if isinstance(nuclide, (int, str)):
        nuclide_list = [nuclide]
    else:
        nuclide_list = list(nuclide)

    zaid_list: List[int] = []
    for n in nuclide_list:
        if isinstance(n, int):
            zaid_list.append(n)
        elif isinstance(n, str):
            zaid_list.append(symbol_to_zaid(n))
        else:
            raise ValueError(f"Invalid nuclide entry: {n!r}")
    mt_list = [mt] if isinstance(mt, int) else list(mt)
    
    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)

    # Apply axis scales
    lx = xscale.lower() if isinstance(xscale, str) else str(xscale)
    ly = yscale.lower() if isinstance(yscale, str) else str(yscale)
    if lx in ("log", "logarithmic"):
        log_x = True
    elif lx in ("lin", "linear"):
        log_x = False
    else:
        raise ValueError(f"Invalid xscale '{xscale}'; expected 'log' or 'linear'")

    if ly in ("log", "logarithmic"):
        log_y = True
    elif ly in ("lin", "linear"):
        log_y = False
    else:
        raise ValueError(f"Invalid yscale '{yscale}'; expected 'log' or 'linear'")

    builder.set_scales(log_x=log_x, log_y=log_y)
    
    # Add uncertainty data for each (zaid, mt) pair
    for z in zaid_list:
        for m in mt_list:
            try:
                # Get uncertainty data using to_plot_data
                _, unc_data = covmat.to_plot_data(zaid=z, mt=m, sigma=sigma, **styling_kwargs)
                
                if unc_data is not None:
                    # Add to plot
                    builder.add_data(unc_data)
            except (ValueError, KeyError) as e:
                # Skip if data not available
                print(f"Warning: Could not plot uncertainties for ZAID={z}, MT={m}: {e}")
                continue
    
    # Set energy range if provided
    if energy_range is not None:
        builder.set_limits(x_lim=(energy_range[0], energy_range[1]))
    
    # Set title behavior:
    # - title == "default": construct a sensible default title
    # - title is None: explicitly omit title
    # - title is a string: use provided string
    if title == "default":
        # Construct default title from nuclide(s) and MT(s)
        try:
            names = [zaid_to_symbol(z) for z in zaid_list]
        except Exception:
            names = [str(z) for z in zaid_list]

        if len(names) == 1:
            nuclide_name = names[0]
        else:
            nuclide_name = ",".join(names)

        if len(mt_list) == 1:
            mt_title = str(mt_list[0])
        else:
            mt_title = ",".join(str(m) for m in mt_list)

        default_title = f"{nuclide_name} Uncertainties MT: {mt_title}"
        builder.set_labels(title=default_title)
    elif title is None:
        # Do not set title (explicitly omit)
        pass
    else:
        builder.set_labels(title=title)

    # Build and configure the plot
    fig = builder.build()

    # Add legend
    if fig.axes:
        ax = fig.axes[0]
        ax.legend(loc=legend_loc)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Relative Uncertainty (%)")

    return fig


def plot_multigroup_xs(
    covmat: CovMat,
    nuclide: Union[int, str, Sequence[Union[int, str]]],
    mt: Union[int, Sequence[int]],
    *,
    energy_range: Optional[Tuple[float, float]] = None,
    show_uncertainties: bool = False,
    sigma: float = 1.0,
    style: str = "light",
    figsize: Tuple[float, float] = (8, 5),
    dpi: int = 300,
    font_family: str = "serif",
    legend_loc: str = "best",
    xscale: str = "log",
    yscale: str = "linear",
    title: Optional[str] = "default",
    **styling_kwargs
) -> plt.Figure:
    """
    Plot multigroup cross sections with optional uncertainty bands.
    
    This modern implementation uses the PlotBuilder infrastructure with to_plot_data()
    for cleaner, more maintainable code.

    Parameters
    ----------
    covmat : CovMat
        The covariance matrix object
    zaid : int or sequence of int
        Isotope ID(s) to plot (e.g., 92235 for U-235)
    mt : int or sequence of int
        Reaction MT number(s) to plot
    energy_range : tuple of float, optional
        Energy range (min, max) for x-axis in MeV. If None, uses full range.
    show_uncertainties : bool, default False
        Whether to show uncertainty bands around cross sections
    sigma : float, default 1.0
        Number of sigma levels for uncertainty bands (e.g., 1.0 for 1σ, 2.0 for 2σ)
    style : str, default "light"
        Plot style: 'light', 'dark', 'paper', 'publication', 'presentation'
    figsize : tuple, default (8, 5)
        Figure size in inches (width, height)
    dpi : int, default 300
        Dots per inch for figure resolution
    font_family : str, default "serif"
        Font family for text elements
    legend_loc : str, default "best"
        Legend location
    **styling_kwargs
        Additional styling arguments (color, linestyle, linewidth, etc.)

    Returns
    -------
    plt.Figure
        The matplotlib figure containing the cross section plots

    Examples
    --------
    Plot cross sections for a single reaction:
    
    >>> fig = plot_multigroup_xs(covmat, zaid=92235, mt=2)
    
    Plot multiple reactions with uncertainty bands:
    
    >>> fig = plot_multigroup_xs(
    ...     covmat, zaid=92235, mt=[2, 18, 102],
    ...     show_uncertainties=True
    ... )

    See Also
    --------
    plot_uncertainties : Plot only uncertainties
    plot_covariance_heatmap : Plot covariance heatmaps
    """
    # Normalize inputs to lists and convert nuclide symbols to ZAID
    from kika._utils import symbol_to_zaid, zaid_to_symbol

    if isinstance(nuclide, (int, str)):
        nuclide_list = [nuclide]
    else:
        nuclide_list = list(nuclide)

    zaid_list: List[int] = []
    for n in nuclide_list:
        if isinstance(n, int):
            zaid_list.append(n)
        elif isinstance(n, str):
            zaid_list.append(symbol_to_zaid(n))
        else:
            raise ValueError(f"Invalid nuclide entry: {n!r}")

    mt_list = [mt] if isinstance(mt, int) else list(mt)

    # Create PlotBuilder
    builder = PlotBuilder(style=style, figsize=figsize, dpi=dpi, font_family=font_family)

    # Apply axis scales
    lx = xscale.lower() if isinstance(xscale, str) else str(xscale)
    ly = yscale.lower() if isinstance(yscale, str) else str(yscale)
    if lx in ("log", "logarithmic"):
        log_x = True
    elif lx in ("lin", "linear"):
        log_x = False
    else:
        raise ValueError(f"Invalid xscale '{xscale}'; expected 'log' or 'linear'")

    if ly in ("log", "logarithmic"):
        log_y = True
    elif ly in ("lin", "linear"):
        log_y = False
    else:
        raise ValueError(f"Invalid yscale '{yscale}'; expected 'log' or 'linear'")

    builder.set_scales(log_x=log_x, log_y=log_y)
    
    # Add cross section data for each (zaid, mt) pair
    for z in zaid_list:
        for m in mt_list:
            try:
                # Get XS and uncertainty data using to_plot_data
                xs_data, unc_data = covmat.to_plot_data(zaid=z, mt=m, sigma=sigma, **styling_kwargs)
                
                if xs_data is not None:
                    # Add to plot with optional uncertainty
                    if show_uncertainties and unc_data is not None:
                        builder.add_data(xs_data, uncertainty=unc_data)
                    else:
                        builder.add_data(xs_data)
            except (ValueError, KeyError) as e:
                # Skip if data not available
                print(f"Warning: Could not plot XS for ZAID={z}, MT={m}: {e}")
                continue
    
    # Set energy range if provided
    if energy_range is not None:
        builder.set_limits(x_lim=(energy_range[0], energy_range[1]))
    
    # Set title behavior (same semantics as uncertainties plot)
    if title == "default":
        try:
            names = [zaid_to_symbol(z) for z in zaid_list]
        except Exception:
            names = [str(z) for z in zaid_list]

        if len(names) == 1:
            nuclide_name = names[0]
        else:
            nuclide_name = ",".join(names)

        if len(mt_list) == 1:
            mt_title = str(mt_list[0])
        else:
            mt_title = ",".join(str(m) for m in mt_list)

        default_title = f"{nuclide_name} Cross Sections MT: {mt_title}"
        builder.set_labels(title=default_title)
    elif title is None:
        pass
    else:
        builder.set_labels(title=title)

    # Build and configure the plot
    fig = builder.build()

    # Add legend and labels
    if fig.axes:
        ax = fig.axes[0]
        ax.legend(loc=legend_loc)
        ax.set_xlabel("Energy (MeV)")
        ax.set_ylabel("Cross Section (barns)")

    return fig
