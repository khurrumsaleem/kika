"""
Angular distribution plotting utilities for EXFOR data.

This module provides functions for plotting experimental angular distributions
and comparing them with theoretical (ACE) data.

Default plotting conventions:
- X-axis: cos(theta) from -1 to 1 (backward to forward angles)
- Y-axis: logarithmic scale for cross sections
"""

from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from kika.exfor.angular_distribution import ExforAngularDistribution


def _get_matplotlib():
    """Lazy import matplotlib."""
    import matplotlib.pyplot as plt
    return plt


def plot_exfor_angular(
    exfor: "ExforAngularDistribution",
    energy: float,
    ax: "plt.Axes" = None,
    tolerance: float = 0.05,
    use_mev: bool = True,
    label: str = None,
    color: str = None,
    marker: str = "o",
    use_cos: bool = True,
    use_log: bool = True,
    **kwargs,
) -> "plt.Axes":
    """
    Plot EXFOR angular distribution at a specific energy.

    Parameters
    ----------
    exfor : ExforAngularDistribution
        EXFOR angular distribution object
    energy : float
        Energy value (in MeV if use_mev=True, else original units)
    ax : plt.Axes, optional
        Matplotlib axes (created if None)
    tolerance : float
        Relative tolerance for energy matching (default: 0.05)
    use_mev : bool
        If True, interpret energy in MeV (default: True)
    label : str, optional
        Plot label (default: citation label)
    color : str, optional
        Line/marker color
    marker : str
        Marker style (default: 'o')
    use_cos : bool
        If True, plot vs cos(theta) from -1 to 1 (default: True)
        If False, plot vs angle in degrees from 0 to 180
    use_log : bool
        If True, use logarithmic y-axis (default: True)
    **kwargs
        Additional arguments passed to errorbar

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    plt = _get_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Get data at energy using to_dataframe
    energy_unit = 'MeV' if use_mev else exfor.units['energy']
    df = exfor.to_dataframe(
        energy=energy,
        tolerance=tolerance,
        energy_unit=energy_unit,
        cross_section_unit='b/sr',
        angle_unit='deg',
    )

    if df.empty:
        available = exfor.energies(unit='MeV') if use_mev else exfor.energies()
        print(f"Warning: No data at E={energy}. Available energies: {available}")
        return ax

    # Determine plot label
    if label is None:
        label = exfor.label

    # Get x-axis values (cos or degrees)
    angles_deg = df["angle"].values
    if use_cos:
        x_values = np.cos(np.radians(angles_deg))
        x_label = r"$\cos(\theta_{" + exfor.angle_frame + r"})$"
        x_lim = (-1, 1)
    else:
        x_values = angles_deg
        x_label = f"Scattering Angle (degrees, {exfor.angle_frame})"
        x_lim = (0, 180)

    # Plot with error bars
    plot_kwargs = dict(
        fmt=marker,
        markersize=5,
        capsize=3,
        alpha=0.8,
        label=label,
    )
    if color is not None:
        plot_kwargs["color"] = color
    plot_kwargs.update(kwargs)

    ax.errorbar(
        x_values,
        df["value"].values,
        yerr=df["error"].values,
        **plot_kwargs,
    )

    # Formatting
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\frac{d\sigma}{d\Omega}$ (b/sr)")
    ax.set_title(f"Angular Distribution at E = {energy} MeV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(*x_lim)

    if use_log:
        ax.set_yscale('log')

    return ax


def plot_exfor_ace_comparison(
    exfor: "ExforAngularDistribution",
    ace_data,
    energy: float,
    mt: int = 2,
    ax: "plt.Axes" = None,
    convert_to_cm: bool = True,
    m_proj: float = None,
    tolerance: float = 0.05,
    use_cos: bool = True,
    use_log: bool = True,
    exfor_kwargs: dict = None,
    ace_kwargs: dict = None,
) -> "plt.Axes":
    """
    Plot EXFOR data overlaid with ACE theoretical data.

    Parameters
    ----------
    exfor : ExforAngularDistribution
        EXFOR angular distribution object
    ace_data
        ACE object with angular distribution
    energy : float
        Energy in MeV
    mt : int
        MT number (2 for elastic scattering, default: 2)
    ax : plt.Axes, optional
        Matplotlib axes (created if None)
    convert_to_cm : bool
        If True, convert EXFOR data to CM frame for comparison (default: True)
    m_proj : float, optional
        Projectile mass in amu (default: neutron)
    tolerance : float
        Energy tolerance for EXFOR data (default: 0.05)
    use_cos : bool
        If True, plot vs cos(theta) from -1 to 1 (default: True)
        If False, plot vs angle in degrees from 0 to 180
    use_log : bool
        If True, use logarithmic y-axis (default: True)
    exfor_kwargs : dict, optional
        Additional kwargs for EXFOR plot
    ace_kwargs : dict, optional
        Additional kwargs for ACE plot

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    from kika.exfor._constants import NEUTRON_MASS_AMU

    plt = _get_matplotlib()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    if m_proj is None:
        m_proj = NEUTRON_MASS_AMU

    # Prepare EXFOR data
    exfor_to_plot = exfor
    if convert_to_cm and exfor.angle_frame == "LAB":
        exfor_to_plot = exfor.convert_to_cm(m_proj=m_proj)

    # Get EXFOR data using to_dataframe
    df = exfor_to_plot.to_dataframe(
        energy=energy,
        tolerance=tolerance,
        energy_unit='MeV',
        cross_section_unit='b/sr',
        angle_unit='deg',
    )

    frame = exfor_to_plot.angle_frame

    if df.empty:
        print(f"Warning: No EXFOR data at E={energy} MeV")
    else:
        # Get x-axis values (cos or degrees)
        angles_deg = df["angle"].values
        if use_cos:
            x_values = np.cos(np.radians(angles_deg))
        else:
            x_values = angles_deg

        # Plot EXFOR data
        exfor_plot_kwargs = dict(
            fmt="o",
            markersize=5,
            capsize=3,
            alpha=0.8,
            label=exfor_to_plot.label,
            color="red",
        )
        if exfor_kwargs:
            exfor_plot_kwargs.update(exfor_kwargs)

        ax.errorbar(
            x_values,
            df["value"].values,
            yerr=df["error"].values,
            **exfor_plot_kwargs,
        )

    # Plot ACE data
    try:
        from kika.exfor.AD_utils import (
            calculate_differential_cross_section,
            cosine_to_angle_degrees,
        )

        cosines, dsigma_domega, sigma_total = calculate_differential_cross_section(ace_data, energy, mt)

        if use_cos:
            x_ace = cosines
        else:
            x_ace = cosine_to_angle_degrees(cosines)

        ace_plot_kwargs = dict(
            linewidth=2,
            label=f"ACE (sigma={sigma_total:.4f} b)",
            color="blue",
        )
        if ace_kwargs:
            ace_plot_kwargs.update(ace_kwargs)

        ax.plot(x_ace, dsigma_domega, **ace_plot_kwargs)
        ax.plot(x_ace, dsigma_domega, "bo", markersize=3)

    except Exception as e:
        print(f"Warning: Could not plot ACE data: {e}")

    # Formatting
    if use_cos:
        ax.set_xlabel(r"$\cos(\theta_{" + frame + r"})$")
        ax.set_xlim(-1, 1)
    else:
        ax.set_xlabel(f"Scattering Angle (degrees, {frame})")
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))

    ax.set_ylabel(r"$\frac{d\sigma}{d\Omega}$ (b/sr)")
    ax.set_title(f"Angular Distribution at E = {energy} MeV (MT={mt})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if use_log:
        ax.set_yscale('log')

    plt.tight_layout()

    return ax


def plot_multiple_energies(
    exfor_list: List["ExforAngularDistribution"],
    energies: List[float],
    ace_data=None,
    mt: int = 2,
    max_plots_per_figure: int = 6,
    figsize: Tuple[int, int] = (15, 10),
    convert_to_cm: bool = True,
    use_cos: bool = True,
    use_log: bool = True,
) -> List["plt.Figure"]:
    """
    Create comparison plots for multiple energies.

    Parameters
    ----------
    exfor_list : List[ExforAngularDistribution]
        List of EXFOR angular distribution objects
    energies : List[float]
        List of energies (MeV) to plot
    ace_data : optional
        Optional ACE object for theoretical comparison
    mt : int
        MT number (2 for elastic, default: 2)
    max_plots_per_figure : int
        Maximum subplots per figure (default: 6)
    figsize : Tuple[int, int]
        Figure size (default: (15, 10))
    convert_to_cm : bool
        Convert experimental data to CM frame (default: True)
    use_cos : bool
        If True, plot vs cos(theta) from -1 to 1 (default: True)
        If False, plot vs angle in degrees from 0 to 180
    use_log : bool
        If True, use logarithmic y-axis (default: True)

    Returns
    -------
    List[plt.Figure]
        List of matplotlib Figure objects
    """
    from kika.exfor._constants import NEUTRON_MASS_AMU

    plt = _get_matplotlib()

    figures = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(exfor_list), 10)))

    # Create figures with subplots
    num_figures = (len(energies) + max_plots_per_figure - 1) // max_plots_per_figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * max_plots_per_figure
        end_idx = min(start_idx + max_plots_per_figure, len(energies))
        energies_this_fig = energies[start_idx:end_idx]

        n_plots = len(energies_this_fig)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_plots > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, energy in enumerate(energies_this_fig):
            ax = axes[i]

            # Plot ACE data if available
            if ace_data is not None:
                try:
                    from kika.exfor.AD_utils import (
                        calculate_differential_cross_section,
                        cosine_to_angle_degrees,
                    )

                    cosines, dsigma_domega, sigma_total = calculate_differential_cross_section(
                        ace_data, energy, mt
                    )

                    if use_cos:
                        x_ace = cosines
                    else:
                        x_ace = cosine_to_angle_degrees(cosines)

                    ax.plot(x_ace, dsigma_domega, "b-", linewidth=2,
                           label=f"ACE (sigma={sigma_total:.3f}b)", zorder=10)
                    ax.plot(x_ace, dsigma_domega, "bo", markersize=3, zorder=10)
                except Exception as e:
                    ax.text(0.5, 0.5, f"ACE error: {e}",
                           transform=ax.transAxes, ha="center", fontsize=8)

            # Plot experimental data
            for j, exfor in enumerate(exfor_list):
                try:
                    exfor_to_plot = exfor
                    if convert_to_cm and exfor.angle_frame == "LAB":
                        exfor_to_plot = exfor.convert_to_cm()

                    # Get data using to_dataframe
                    df = exfor_to_plot.to_dataframe(
                        energy=energy,
                        tolerance=0.1,
                        energy_unit='MeV',
                        cross_section_unit='b/sr',
                        angle_unit='deg',
                    )

                    if df.empty:
                        continue

                    # Get x-axis values
                    angles_deg = df["angle"].values
                    if use_cos:
                        x_values = np.cos(np.radians(angles_deg))
                    else:
                        x_values = angles_deg

                    ax.errorbar(
                        x_values,
                        df["value"].values,
                        yerr=df["error"].values,
                        fmt="o",
                        color=colors[j % len(colors)],
                        markersize=4,
                        capsize=3,
                        label=exfor.label,
                        alpha=0.8,
                    )
                except Exception:
                    continue

            # Formatting
            ax.set_title(f"E = {energy} MeV", fontsize=12)
            if use_cos:
                ax.set_xlabel(r"$\cos(\theta_{CM})$", fontsize=10)
                ax.set_xlim(-1, 1)
            else:
                ax.set_xlabel("Angle (degrees, CM)", fontsize=10)
                ax.set_xlim(0, 180)
            ax.set_ylabel("dsigma/dOmega (b/sr)", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            if use_log:
                ax.set_yscale('log')

        # Remove empty subplots
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        figures.append(fig)

    return figures
