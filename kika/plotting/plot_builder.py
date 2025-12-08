"""
PlotBuilder class for composing and rendering plots.

This module provides the PlotBuilder class that takes PlotData objects
and creates publication-quality plots with consistent styling.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from .plot_data import (
    PlotData,
    UncertaintyBand,
    LegendreUncertaintyPlotData,
    MultigroupUncertaintyPlotData,
    HeatmapPlotData,
    CovarianceHeatmapData,
    MF34HeatmapData
)
from .styles import (
    _get_color_palette,
    _get_linestyles,
    _apply_style_to_rcparams,
    _adjust_figsize_for_notebook,
    _adjust_dpi_for_notebook,
    format_energy_axis_ticks
)
from ._backend_utils import (
    _is_notebook,
    _detect_interactive_backend,
    _setup_notebook_backend,
    _configure_figure_interactivity
)


class PlotBuilder:
    """
    Builder class for creating plots from PlotData objects.
    
    This class handles the composition of multiple plot elements and applies
    consistent styling and formatting.
    
    Examples
    --------
    >>> # Create plot data objects
    >>> data1 = LegendreCoeffPlotData(x=energies1, y=coeffs1, order=1, isotope='U235')
    >>> data2 = LegendreCoeffPlotData(x=energies2, y=coeffs2, order=1, isotope='U238')
    >>> 
    >>> # Build the plot
    >>> builder = PlotBuilder(style='light', figsize=(10, 6))
    >>> builder.add_data(data1, color='blue')
    >>> builder.add_data(data2, color='red')
    >>> builder.set_labels(
    ...     title='Elastic Scattering Legendre Coefficients',
    ...     x_label='Energy (eV)',
    ...     y_label='Coefficient Value'
    ... )
    >>> fig = builder.build()
    >>> fig.show()
    """
    
    def __init__(
        self,
        style: str = 'light',
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 100,
        ax: Optional[plt.Axes] = None,
        projection: Optional[str] = None,
        font_family: str = 'serif',
        notebook_mode: Optional[bool] = None,
        interactive: Optional[bool] = None,
    ):
        """
        Initialize the PlotBuilder.
        
        Parameters
        ----------
        style : str
            Plot style: 'light' (default, publication-quality) or 'dark'
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Dots per inch for figure resolution
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure and axes.
        projection : str, optional
            Projection type (e.g., '3d' for 3D plots)
        font_family : str
            Font family for text elements (default: 'serif')
        notebook_mode : bool, optional
            Force notebook mode (auto-detected if None)
        interactive : bool, optional
            Force interactive mode (auto-detected if None)
        """
        if style not in ('light', 'dark'):
            raise ValueError(f"Invalid style '{style}'. Must be 'light' or 'dark'.")
        
        self.style = style
        self.projection = projection
        self.font_family = font_family
        
        # Auto-detect notebook and interactive mode
        if notebook_mode is None:
            notebook_mode = _is_notebook()
        if interactive is None:
            interactive = notebook_mode and _detect_interactive_backend()
        
        self._notebook_mode = notebook_mode
        self._interactive = interactive
        
        # Adjust settings for notebook environment
        # Keep a record of the user-requested size/dpi before any notebook tweaks
        self._figsize_user = figsize
        self._dpi_user = dpi

        if notebook_mode:
            figsize = _adjust_figsize_for_notebook(figsize, interactive)
            dpi = _adjust_dpi_for_notebook(dpi, interactive)
            
            # Setup interactive backend if needed
            if interactive:
                backend_success, backend_msg = _setup_notebook_backend()
                self._backend_info = backend_msg
                self._interactive = backend_success
            else:
                self._backend_info = f"Notebook mode, backend: {plt.get_backend()}"
        else:
            self._backend_info = f"Non-notebook environment, backend: {plt.get_backend()}"
        
        self.figsize = figsize
        self.dpi = dpi
        
        # Get color palette and linestyles for this style
        self._colors = _get_color_palette(style)
        self._linestyles = _get_linestyles()
        
        # Storage for plot elements
        self._data_list: List[PlotData] = []
        self._uncertainty_bands: List[Tuple[UncertaintyBand, int]] = []  # (band, data_index)
        self._custom_styling: List[Dict[str, Any]] = []  # Per-data styling overrides
        
        # Storage for heatmap data (alternative to line plots)
        self._heatmap_data: Optional['HeatmapPlotData'] = None
        self._heatmap_show_uncertainties: bool = True
        self._heatmap_styling_overrides: Dict[str, Any] = {}
        
        # Plot configuration
        self._x_label: Optional[str] = None
        self._y_label: Optional[str] = None
        self._title: Optional[str] = None
        self._legend_loc: str = 'best'
        self._use_log_x: bool = False
        self._use_log_y: bool = False
        self._x_lim: Optional[Tuple[float, float]] = None
        self._y_lim: Optional[Tuple[float, float]] = None
        self._grid: bool = True
        self._grid_alpha: float = 0.3  # Alpha (transparency) for major grid
        self._show_minor_grid: bool = False  # Whether to show minor grid
        self._minor_grid_alpha: float = 0.15  # Alpha for minor grid
        
        # Font size configuration (will override style defaults if set)
        self._title_fontsize: Optional[float] = None
        self._label_fontsize: Optional[float] = None
        self._tick_labelsize: Optional[float] = None
        self._legend_fontsize: Optional[float] = None
        
        # Setup figure and axes
        if ax is not None:
            # Use provided axes
            self.fig = ax.figure
            self.ax = ax
        else:
            # Apply style to matplotlib rcParams
            _apply_style_to_rcparams(
                style=style,
                notebook_mode=notebook_mode,
                figsize=figsize,
                dpi=dpi,
                font_family=font_family,
                projection=projection
            )
            
            # Create figure and axes
            if projection is not None:
                self.fig = plt.figure(figsize=figsize, dpi=dpi)
                self.ax = self.fig.add_subplot(111, projection=projection)
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Configure figure interactivity
            _configure_figure_interactivity(self.fig, interactive)
    
    def add_data(
        self,
        data: Union[PlotData, Tuple[PlotData, Optional[Union[UncertaintyBand, PlotData]]]],
        uncertainty: Optional[Union[UncertaintyBand, PlotData]] = None,
        **styling_overrides
    ) -> 'PlotBuilder':
        """
        Add a PlotData object to the plot.
        
        Parameters
        ----------
        data : PlotData or tuple
            The data to plot. Can be either:
            - A PlotData object
            - A tuple (PlotData, UncertaintyBand) as returned by to_plot_data with uncertainty=True
            If a tuple is provided and uncertainty is None, the UncertaintyBand from the tuple will be used.
        uncertainty : UncertaintyBand or PlotData, optional
            Uncertainty to plot with this data. Can be either:
            - UncertaintyBand object (will be plotted as shaded region)
            - PlotData object with uncertainty values (will be converted to band)
            If data is a tuple and this parameter is provided, this parameter takes precedence.
        **styling_overrides
            Styling overrides for this specific data (color, linestyle, etc.)
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        # Check for conflicts with heatmap data
        if self._heatmap_data is not None:
            raise ValueError(
                "Cannot mix line plots and heatmaps. "
                "add_data() cannot be called after add_heatmap(). "
                "Create separate PlotBuilder instances for heatmaps and line plots."
            )
        
        # Handle tuple input from to_plot_data(uncertainty=True)
        if isinstance(data, tuple):
            if len(data) == 2:
                plot_data, tuple_uncertainty = data
                # Use the uncertainty from tuple if no explicit uncertainty is provided
                if uncertainty is None:
                    uncertainty = tuple_uncertainty
                data = plot_data
            else:
                raise ValueError(f"Expected tuple of length 2, got {len(data)}")
        
        self._data_list.append(data)
        self._custom_styling.append(styling_overrides)
        
        if uncertainty is not None:
            # Convert PlotData to UncertaintyBand if needed
            if isinstance(uncertainty, PlotData):
                uncertainty = self._convert_plotdata_to_band(uncertainty, data)
            
            data_index = len(self._data_list) - 1
            self._uncertainty_bands.append((uncertainty, data_index))
        
        return self
    
    def add_multiple(
        self,
        data_list: List[PlotData],
        colors: Optional[List[Union[str, Tuple]]] = None,
        linestyles: Optional[List[str]] = None,
        **common_styling
    ) -> 'PlotBuilder':
        """
        Add multiple PlotData objects at once.
        
        Parameters
        ----------
        data_list : list of PlotData
            List of data objects to plot
        colors : list of str/tuple, optional
            Colors for each data object. If None, uses automatic color cycling.
        linestyles : list of str, optional
            Line styles for each data object. If None, uses solid lines.
        **common_styling
            Styling applied to all data objects
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        for i, data in enumerate(data_list):
            styling = common_styling.copy()
            
            if colors is not None and i < len(colors):
                styling['color'] = colors[i]
            if linestyles is not None and i < len(linestyles):
                styling['linestyle'] = linestyles[i]
            
            self.add_data(data, **styling)
        
        return self
    
    def set_labels(
        self,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> 'PlotBuilder':
        """
        Set plot labels.
        
        Parameters
        ----------
        title : str, optional
            Plot title
        x_label : str, optional
            X-axis label
        y_label : str, optional
            Y-axis label
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        if title is not None:
            self._title = title
        if x_label is not None:
            self._x_label = x_label
        if y_label is not None:
            self._y_label = y_label
        
        return self
    
    def set_scales(
        self,
        log_x: bool = False,
        log_y: bool = False
    ) -> 'PlotBuilder':
        """
        Set axis scales.
        
        Parameters
        ----------
        log_x : bool
            Use logarithmic scale for x-axis
        log_y : bool
            Use logarithmic scale for y-axis
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._use_log_x = log_x
        self._use_log_y = log_y
        return self
    
    def set_limits(
        self,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None
    ) -> 'PlotBuilder':
        """
        Set axis limits.
        
        Parameters
        ----------
        x_lim : tuple, optional
            (min, max) for x-axis
        y_lim : tuple, optional
            (min, max) for y-axis
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._x_lim = x_lim
        self._y_lim = y_lim
        return self
    
    def set_legend(self, loc: str = 'best') -> 'PlotBuilder':
        """
        Set legend location.
        
        Parameters
        ----------
        loc : str
            Legend location
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._legend_loc = loc
        return self
    
    def set_grid(
        self, 
        grid: bool = True,
        alpha: float = 0.3,
        show_minor: bool = False,
        minor_alpha: float = 0.15
    ) -> 'PlotBuilder':
        """
        Configure grid display settings.
        
        Parameters
        ----------
        grid : bool
            Whether to show major grid
        alpha : float
            Alpha (transparency) for major grid lines. Range: 0.0-1.0
        show_minor : bool
            Whether to show minor grid lines
        minor_alpha : float
            Alpha (transparency) for minor grid lines. Range: 0.0-1.0
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._grid = grid
        self._grid_alpha = alpha
        self._show_minor_grid = show_minor
        self._minor_grid_alpha = minor_alpha
        return self
    
    def set_tick_params(
        self,
        max_ticks_x: Optional[int] = None,
        max_ticks_y: Optional[int] = None,
        rotate_x: Optional[float] = None,
        rotate_y: Optional[float] = None,
        auto_rotate: bool = False
    ) -> 'PlotBuilder':
        """
        Configure tick parameters to avoid overlapping labels.
        
        This method helps prevent tick label overlap, especially useful for
        small figures or dense data.
        
        Parameters
        ----------
        max_ticks_x : int, optional
            Maximum number of ticks on x-axis. If None, matplotlib default.
        max_ticks_y : int, optional
            Maximum number of ticks on y-axis. If None, matplotlib default.
        rotate_x : float, optional
            Rotation angle for x-axis tick labels (degrees). 
            Common values: 45, 90 for rotated text.
        rotate_y : float, optional
            Rotation angle for y-axis tick labels (degrees).
        auto_rotate : bool, optional
            If True, automatically rotate x-axis labels by 45° for better spacing.
            Default is False.
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> # Limit to 8 ticks on x-axis and rotate labels
        >>> builder.set_tick_params(max_ticks_x=8, rotate_x=45)
        >>> 
        >>> # Auto-rotate x-axis labels for better spacing
        >>> builder.set_tick_params(auto_rotate=True)
        >>> 
        >>> # Control both axes
        >>> builder.set_tick_params(max_ticks_x=6, max_ticks_y=8)
        """
        # Store tick parameters
        if not hasattr(self, '_tick_params'):
            self._tick_params = {}
        
        if max_ticks_x is not None:
            self._tick_params['max_ticks_x'] = max_ticks_x
        if max_ticks_y is not None:
            self._tick_params['max_ticks_y'] = max_ticks_y
        if rotate_x is not None:
            self._tick_params['rotate_x'] = rotate_x
        if rotate_y is not None:
            self._tick_params['rotate_y'] = rotate_y
        if auto_rotate:
            self._tick_params['rotate_x'] = 45
        
        return self
    
    def set_font_sizes(
        self,
        title: Optional[float] = None,
        labels: Optional[float] = None,
        ticks: Optional[float] = None,
        legend: Optional[float] = None
    ) -> 'PlotBuilder':
        """
        Set font sizes for plot elements.
        
        This method allows fine-grained control over font sizes, overriding
        the defaults set by the style parameter.
        
        Parameters
        ----------
        title : float, optional
            Font size for the plot title
        labels : float, optional
            Font size for axis labels (x and y)
        ticks : float, optional
            Font size for tick labels
        legend : float, optional
            Font size for legend text
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Examples
        --------
        >>> builder = PlotBuilder(style='paper')
        >>> builder.add_data(data)
        >>> builder.set_labels(title='My Plot', x_label='Energy', y_label='Value')
        >>> builder.set_font_sizes(title=18, labels=14, ticks=12, legend=11)
        >>> fig = builder.build()
        """
        if title is not None:
            self._title_fontsize = title
        if labels is not None:
            self._label_fontsize = labels
        if ticks is not None:
            self._tick_labelsize = ticks
        if legend is not None:
            self._legend_fontsize = legend
        
        return self
    
    def add_heatmap(
        self,
        heatmap_data: 'HeatmapPlotData',
        show_uncertainties: bool = True,
        **styling_overrides
    ) -> 'PlotBuilder':
        """
        Add heatmap data to be rendered when build() is called.
        
        This method stores heatmap data for rendering, following the builder pattern
        consistently with add_data(). Call build() to generate the figure.
        
        Parameters
        ----------
        heatmap_data : HeatmapPlotData
            Heatmap data object (CovarianceHeatmapData or MF34HeatmapData)
        show_uncertainties : bool, default True
            Whether to show uncertainty panels above the heatmap
        **styling_overrides
            Styling overrides for heatmap rendering. Common options:
            - cmap : str or Colormap - Override colormap (e.g., 'viridis', 'RdBu_r')
            - vmin : float - Minimum value for colormap normalization
            - vmax : float - Maximum value for colormap normalization
            - norm : Normalize - Custom normalization (overrides vmin/vmax)
            - colorbar_label : str - Override colorbar label
            
        Returns
        -------
        PlotBuilder
            Self for method chaining
            
        Notes
        -----
        Heatmaps and line plots are mutually exclusive. You cannot mix add_data() and
        add_heatmap() on the same PlotBuilder instance. Calling add_heatmap() after
        add_data() (or vice versa) will raise a ValueError.
        
        Calling add_heatmap() multiple times will overwrite the previous heatmap data.
        
        Examples
        --------
        >>> # Basic usage
        >>> from kika.plotting import PlotBuilder
        >>> heatmap_data = covmat.to_heatmap_data(nuclide=92235, mt=[2, 18])
        >>> builder = PlotBuilder(style='light')
        >>> builder.add_heatmap(heatmap_data)
        >>> builder.set_labels(title='Custom Title')
        >>> fig = builder.build()
        
        >>> # With method chaining
        >>> fig = (PlotBuilder(style='light')
        ...        .add_heatmap(heatmap_data, cmap='plasma', show_uncertainties=False)
        ...        .set_labels(title='Correlation Matrix', x_label='MT', y_label='MT')
        ...        .build())
        """
        # Check for conflicts with line plot data
        if self._data_list:
            raise ValueError(
                "Cannot mix line plots and heatmaps. "
                "add_heatmap() cannot be called after add_data(). "
                "Create separate PlotBuilder instances for heatmaps and line plots."
            )
        
        # Store heatmap data for rendering in build()
        self._heatmap_data = heatmap_data
        self._heatmap_show_uncertainties = show_uncertainties
        self._heatmap_styling_overrides = styling_overrides
        
        return self
    
    def _build_heatmap(self) -> plt.Figure:
        """
        Internal method to render heatmap. Called by build() when heatmap data is present.
        
        Returns
        -------
        matplotlib.figure.Figure
            The completed heatmap figure
        """
        # Get stored heatmap data
        heatmap_data = self._heatmap_data
        show_uncertainties = self._heatmap_show_uncertainties
        styling_overrides = self._heatmap_styling_overrides
        
        from .plot_data import CovarianceHeatmapData, MF34HeatmapData, HeatmapPlotData
        from .heatmap_utils import (
            setup_energy_group_ticks,
            setup_energy_group_ticks_single_block,
            format_uncertainty_ticks,
            add_mt_labels_to_heatmap
        )
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm
        
        # For heatmaps, use manual formatting instead of style system
        # The old implementation did this to avoid issues with heatmap tick/label positioning
        plt.rcdefaults()
        figsize = getattr(self, "_figsize_user", None) or self.figsize
        dpi = getattr(self, "_dpi_user", None) or self.dpi

        plt.rcParams.update({
            'font.family': self.font_family,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2.2,
            'lines.markersize': 7,
            'axes.grid': False,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'figure.constrained_layout.use': False,
        })

        # Close any pre-existing figure created during __init__ to avoid blank displays in notebooks
        if hasattr(self, "fig") and getattr(self, "ax", None) is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig = None
            self.ax = None
        
        # Determine if we have uncertainty data to show
        has_uncertainties = (
            show_uncertainties and
            hasattr(heatmap_data, 'uncertainty_data') and
            heatmap_data.uncertainty_data is not None and
            len(heatmap_data.uncertainty_data) > 0
        )

        # Pre-compute helper to translate energy limits into heatmap axes coordinates
        symmetric_matrix = isinstance(heatmap_data, (CovarianceHeatmapData, MF34HeatmapData)) \
            and getattr(heatmap_data, "is_diagonal", False)
        x_edges_for_limits = getattr(heatmap_data, "x_edges", None)
        y_edges_for_limits = getattr(heatmap_data, "y_edges", None)
        extent = getattr(heatmap_data, "extent", None)
        if x_edges_for_limits is None and extent is not None:
            x_edges_for_limits = np.asarray(extent[:2], dtype=float)
        if y_edges_for_limits is None and extent is not None:
            y_edges_for_limits = np.asarray(extent[2:], dtype=float)

        def _transform_energy_value(val: float) -> float:
            """Map raw energy values onto the transformed axis used by the heatmap."""
            if getattr(heatmap_data, "scale", None) == "log":
                return float(np.log10(max(float(val), 1e-300)))
            return float(val)

        def _convert_limits_to_axis(
            lim: Optional[Tuple[float, float]],
            axis_edges: Optional[np.ndarray]
        ) -> Optional[Tuple[float, float]]:
            """Convert user-provided energy limits into the heatmap axis coordinates."""
            if lim is None or axis_edges is None:
                return lim

            energy_grid = getattr(heatmap_data, "energy_grid", None)
            if energy_grid is None or len(energy_grid) == 0:
                return lim

            axis_edges = np.asarray(axis_edges, dtype=float)
            energy_edges = np.asarray(energy_grid, dtype=float)

            transformed_edges = np.asarray([_transform_energy_value(v) for v in energy_edges], dtype=float)
            base_span = transformed_edges[-1] - transformed_edges[0]
            axis_span = axis_edges[-1] - axis_edges[0]

            # Only attempt conversion when the axis spans a single block (no concatenated MT/L sections)
            if not (np.isfinite(base_span) and np.isfinite(axis_span)) or base_span <= 0 or axis_span <= 0:
                return lim
            if axis_span > base_span * 1.05:
                return lim

            offset = axis_edges[0]
            base0 = transformed_edges[0]
            return (
                _transform_energy_value(lim[0]) - base0 + offset,
                _transform_energy_value(lim[1]) - base0 + offset,
            )

        resolved_x_lim = _convert_limits_to_axis(getattr(self, "_x_lim", None), x_edges_for_limits)
        resolved_y_lim = _convert_limits_to_axis(getattr(self, "_y_lim", None), y_edges_for_limits)
        if symmetric_matrix and resolved_y_lim is None and resolved_x_lim is not None:
            resolved_y_lim = resolved_x_lim
        
        # Create figure and layout
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        if has_uncertainties:
            # GridSpec with 2 rows: uncertainty panels (top) + heatmap (bottom)
            num_panels = len(heatmap_data.uncertainty_data)
            # Adjust figure height to accommodate uncertainty panels (multiply by 1.18)
            fig.set_size_inches(figsize[0], figsize[1] * 1.18)
            gs = GridSpec(2, num_panels if num_panels > 1 else 1, figure=fig,
                         height_ratios=[0.16, 1], hspace=0.08, wspace=0.02)
            
            # Create uncertainty axes
            uncertainty_axes = []
            if num_panels == 1:
                ax_unc = fig.add_subplot(gs[0, :])
                uncertainty_axes.append(ax_unc)
            else:
                for i in range(num_panels):
                    ax_unc = fig.add_subplot(gs[0, i])
                    uncertainty_axes.append(ax_unc)
            
            # Create heatmap axes (spans all columns)
            ax_heatmap = fig.add_subplot(gs[1, :])
        else:
            # Single axes for heatmap only
            ax_heatmap = fig.add_subplot(111)
            uncertainty_axes = None
        
        # Get masked data
        if hasattr(heatmap_data, 'get_masked_data'):
            M = heatmap_data.get_masked_data()
        else:
            M = heatmap_data.matrix_data
        
        # Set background color for masked regions (fixed lightgray)
        ax_heatmap.set_facecolor("#F0F0F0")
        ax_heatmap.grid(False, which="both")
        
        # Apply styling overrides to heatmap_data attributes (create copy to avoid mutating original)
        # Priority: styling_overrides > heatmap_data attributes > defaults
        effective_cmap = styling_overrides.get('cmap', heatmap_data.cmap)
        effective_vmin = styling_overrides.get('vmin', heatmap_data.vmin)
        effective_vmax = styling_overrides.get('vmax', heatmap_data.vmax)
        effective_norm = styling_overrides.get('norm', heatmap_data.norm)
        effective_colorbar_label = styling_overrides.get('colorbar_label', heatmap_data.colorbar_label)
        
        # Setup colormap
        if isinstance(effective_cmap, str):
            cmap = plt.get_cmap(effective_cmap).copy()
        else:
            cmap = effective_cmap
        
        if hasattr(cmap, 'set_bad'):
            cmap.set_bad(color="#F0F0F0")
        
        # Handle normalization
        if effective_norm is not None:
            norm = effective_norm
        elif effective_vmin is not None and effective_vmax is not None:
            norm = plt.Normalize(vmin=effective_vmin, vmax=effective_vmax)
        else:
            # Auto-determine normalization
            vmin = np.nanmin(M)
            vmax = np.nanmax(M)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        # Draw heatmap (respect energy edges/extents when provided)
        if getattr(heatmap_data, "x_edges", None) is not None and getattr(heatmap_data, "y_edges", None) is not None:
            X, Y = np.meshgrid(heatmap_data.x_edges, heatmap_data.y_edges)
            im = ax_heatmap.pcolormesh(X, Y, M, cmap=cmap, norm=norm, shading="flat")
            
            # Set default limits from data
            default_xlim = (heatmap_data.x_edges[0], heatmap_data.x_edges[-1])
            default_ylim = (heatmap_data.y_edges[0], heatmap_data.y_edges[-1])

            x_limits = resolved_x_lim if resolved_x_lim is not None else default_xlim
            y_limits = resolved_y_lim if resolved_y_lim is not None else default_ylim

            ax_heatmap.set_xlim(x_limits)
            ax_heatmap.set_ylim(y_limits)
            ax_heatmap.invert_yaxis()
        elif getattr(heatmap_data, "extent", None) is not None:
            im = ax_heatmap.imshow(
                M,
                cmap=cmap,
                norm=norm,
                origin="upper",
                interpolation="nearest",
                aspect="auto",
                extent=heatmap_data.extent,
            )
            
            default_xlim = heatmap_data.extent[:2]
            default_ylim = heatmap_data.extent[2:]
            x_limits = resolved_x_lim if resolved_x_lim is not None else default_xlim
            y_limits = resolved_y_lim if resolved_y_lim is not None else default_ylim

            ax_heatmap.set_xlim(x_limits)
            ax_heatmap.set_ylim(y_limits)
        else:
            im = ax_heatmap.imshow(M, cmap=cmap, norm=norm,
                                  origin="upper", interpolation="nearest", aspect="auto")
            
            # Apply user-specified limits if provided
            if resolved_x_lim is not None:
                ax_heatmap.set_xlim(resolved_x_lim)
            if resolved_y_lim is not None:
                ax_heatmap.set_ylim(resolved_y_lim)
            elif symmetric_matrix and resolved_x_lim is not None:
                ax_heatmap.set_ylim(resolved_x_lim)
        
        # Handle block structure and ticks
        if isinstance(heatmap_data, CovarianceHeatmapData):
            self._setup_covariance_heatmap_ticks(ax_heatmap, heatmap_data)
        elif isinstance(heatmap_data, MF34HeatmapData):
            self._setup_mf34_heatmap_ticks(ax_heatmap, heatmap_data)

        # Draw grid lines separating MT/L blocks when present
        self._draw_block_boundaries(ax_heatmap, heatmap_data)
        
        # Set axis labels using builder configuration when available
        default_axis_label = "MT number" if isinstance(heatmap_data, CovarianceHeatmapData) else "Legendre L"
        x_label = getattr(self, "_x_label", None) or default_axis_label
        y_label = getattr(self, "_y_label", None) or default_axis_label
        if x_label:
            if self._label_fontsize is not None:
                ax_heatmap.set_xlabel(x_label, fontsize=self._label_fontsize)
            else:
                ax_heatmap.set_xlabel(x_label)
        if y_label:
            if self._label_fontsize is not None:
                ax_heatmap.set_ylabel(y_label, fontsize=self._label_fontsize)
            else:
                ax_heatmap.set_ylabel(y_label)
        
        # Set title (suppress for paper/publication styles) preferring builder title
        effective_title = getattr(self, "_title", None) or getattr(heatmap_data, "label", None)
        if self.style not in ('paper', 'publication') and effective_title:
            title_kwargs = {}
            if self._title_fontsize is not None:
                title_kwargs["fontsize"] = self._title_fontsize
            fig.suptitle(effective_title, y=0.985 if not has_uncertainties else 0.97, **title_kwargs)

        if self._tick_labelsize is not None:
            ax_heatmap.tick_params(axis='both', which='both', labelsize=self._tick_labelsize)
        
        # Draw uncertainty panels if requested
        if has_uncertainties and uncertainty_axes is not None:
            self._draw_uncertainty_panels(uncertainty_axes, heatmap_data, ax_heatmap)
        
        # Adjust layout to make room for colorbar
        fig.canvas.draw()
        heatmap_pos = ax_heatmap.get_position()
        
        # Calculate number of MTs/blocks for layout adjustment
        if isinstance(heatmap_data, CovarianceHeatmapData) and heatmap_data.block_info:
            num_blocks = len(heatmap_data.block_info.get('mts', [1]))
        elif isinstance(heatmap_data, MF34HeatmapData) and heatmap_data.block_info:
            num_blocks = len(heatmap_data.legendre_coeffs)
        else:
            num_blocks = 1
        
        # Bottom margin configuration (expand mildly with number of blocks)
        extra_margin = max(0, num_blocks - 1) * 0.015
        bottom_margin = min(0.12 + extra_margin, 0.26)
        
        if has_uncertainties:
            fig.subplots_adjust(left=0.12, right=0.94, bottom=bottom_margin, top=0.90)
        else:
            fig.subplots_adjust(left=0.12, right=0.94, bottom=bottom_margin, top=0.93)
        
        # Add colorbar (offset increased to 0.10 to avoid overlap with right energy labels)
        fig.canvas.draw()  # Draw to get accurate position
        heatmap_pos = ax_heatmap.get_position()
        cbar_ax = fig.add_axes([
            heatmap_pos.x1 + 0.10,  # Right of heatmap + offset
            heatmap_pos.y0,          # Aligned bottom
            0.03,                    # Width
            heatmap_pos.height       # Full height
        ])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if effective_colorbar_label:
            cbar.set_label(effective_colorbar_label)
        
        # Configure figure interactivity
        _configure_figure_interactivity(fig, self._interactive)
        return fig
    
    def _setup_covariance_heatmap_ticks(self, ax: plt.Axes, data: 'CovarianceHeatmapData') -> None:
        """Setup ticks for covariance heatmap using energy-based coordinates."""
        import numpy as np

        if not data.block_info or data.energy_grid is None:
            return

        tick_grey = "#707070"
        mts = data.block_info.get('mts', [])
        energy_ranges = data.block_info.get('energy_ranges', {}) or {}
        ranges_fallback = data.block_info.get('ranges', []) or []
        mt_labels = data.mt_labels or [str(m) for m in mts]

        edges_raw = np.asarray(data.energy_grid, dtype=float)

        def _fmt_sci_eV(v: float) -> str:
            return f"{v:.0e}"

        def _log_ticks_with_decades(edges_block: np.ndarray):
            Emin = float(edges_block[0])
            Emax = float(edges_block[-1])
            kmin = int(np.ceil(np.log10(max(Emin, 1e-300))))
            kmax = int(np.floor(np.log10(max(Emax, 1e-300))))
            ks = np.arange(kmin, kmax + 1, dtype=int) if kmax >= kmin else np.array([kmin])

            vals = []
            labels = []

            is_emin_decade = np.isclose(Emin, 10.0 ** np.round(np.log10(Emin)), rtol=1e-10)
            is_emax_decade = np.isclose(Emax, 10.0 ** np.round(np.log10(Emax)), rtol=1e-10)
            if not is_emin_decade:
                vals.append(Emin)
                labels.append(_fmt_sci_eV(Emin))

            for k in ks:
                base = 10.0 ** k
                for m in range(1, 10):
                    v = m * base
                    if Emin <= v <= Emax:
                        vals.append(v)
                        labels.append(_fmt_sci_eV(v) if m == 1 else "")

            if not is_emax_decade and (len(vals) == 0 or not np.isclose(vals[-1], Emax, rtol=1e-10)):
                vals.append(Emax)
                labels.append(_fmt_sci_eV(Emax))

            vals_arr = np.array(vals, dtype=float)
            if vals_arr.size == 0:
                return np.array([]), []

            base0 = np.log10(max(Emin, 1e-300))
            return np.log10(vals_arr) - base0, labels

        def _linear_ticks_20pct(edges_block: np.ndarray):
            Emin = float(edges_block[0])
            Emax = float(edges_block[-1])
            if Emax <= Emin:
                return np.array([]), []
            fracs = np.linspace(0.0, 1.0, 6)
            vals = Emin + fracs * (Emax - Emin)
            labels = [_fmt_sci_eV(v) for v in vals]
            return vals - Emin, labels

        def _energy_ticks(edges_block: np.ndarray):
            if data.scale == "log" and edges_block[0] > 0:
                return _log_ticks_with_decades(edges_block)
            return _linear_ticks_20pct(edges_block)

        # Build block coordinate ranges
        x_ranges = []
        if data.is_diagonal:
            for i, m in enumerate(mts):
                xr = energy_ranges.get(m)
                if xr is None and len(ranges_fallback) > i:
                    xr = ranges_fallback[i]
                if xr is not None:
                    x_ranges.append(tuple(xr))
        else:
            # off-diagonal: first mt is row, second is col
            if len(mts) >= 2:
                row_mt, col_mt = mts[0], mts[1]
                y_range = energy_ranges.get(row_mt) or (ranges_fallback[0] if ranges_fallback else None)
                x_range = energy_ranges.get(col_mt) or (ranges_fallback[1] if len(ranges_fallback) > 1 else None)
                if x_range is not None:
                    x_ranges.append(tuple(x_range))
                if y_range is not None:
                    y_range_val = tuple(y_range)
                else:
                    y_range_val = None
            else:
                x_range = ranges_fallback[0] if ranges_fallback else None
                y_range_val = ranges_fallback[0] if ranges_fallback else None
                if x_range is not None:
                    x_ranges.append(tuple(x_range))

        # MT labels on primary axes using block centers
        if data.is_diagonal and x_ranges:
            centers = [(a + b) * 0.5 for (a, b) in x_ranges]
            ax.set_xticks(centers)
            ax.set_xticklabels(mt_labels)
            ax.set_yticks(centers)
            ax.set_yticklabels(mt_labels)
        elif not data.is_diagonal and len(mts) >= 2:
            if x_ranges:
                cx = (x_ranges[0][0] + x_ranges[0][1]) * 0.5
                ax.set_xticks([cx])
                ax.set_xticklabels([mt_labels[1] if len(mt_labels) > 1 else str(mts[1])])
            if 'y_range_val' in locals() and y_range_val is not None:
                cy = (y_range_val[0] + y_range_val[1]) * 0.5
                ax.set_yticks([cy])
                ax.set_yticklabels([mt_labels[0] if mt_labels else str(mts[0])])

        if not data.show_energy_ticks or x_ranges == []:
            return

        pos_local, labels_all = _energy_ticks(edges_raw)

        if data.is_diagonal and len(x_ranges) > 0:
            top_ticks = []
            top_labels = []
            side_ticks = []
            side_labels = []
            for i, xr in enumerate(x_ranges):
                pos_global = pos_local + xr[0]
                filtered_top_labels = []
                for lbl in labels_all:
                    if lbl and (lbl == labels_all[0] or lbl == labels_all[-1]):
                        filtered_top_labels.append("")
                    else:
                        filtered_top_labels.append(lbl)
                top_ticks.extend(pos_global.tolist())
                top_labels.extend(filtered_top_labels)

                is_last_block = (i == len(x_ranges) - 1)
                side_ticks.extend(pos_global.tolist())
                filtered_side_labels = []
                for lbl in labels_all:
                    if lbl and lbl == labels_all[-1]:
                        filtered_side_labels.append(lbl if is_last_block else "")
                    else:
                        filtered_side_labels.append(lbl)
                side_labels.extend(filtered_side_labels)
        else:
            x_range = x_ranges[0] if x_ranges else (0.0, 0.0)
            pos_global = pos_local + x_range[0]
            top_ticks = pos_global.tolist()
            side_ticks = pos_global.tolist()
            top_labels = []
            for lbl in labels_all:
                if lbl and (lbl == labels_all[0] or lbl == labels_all[-1]):
                    top_labels.append("")
                else:
                    top_labels.append(lbl)
            side_labels = labels_all

        ax_top = ax.twiny()
        ax_right = ax.twinx()
        ax_bottom = ax.twiny()
        ax_left = ax.twinx()

        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels(top_labels, fontsize=9, color=tick_grey)
        ax_top.tick_params(axis='x', direction='out', length=3, colors=tick_grey, pad=2, top=True, bottom=False)
        ax_top.spines['top'].set_visible(True)
        ax_top.spines['top'].set_linewidth(0.6)
        ax_top.spines['top'].set_color(tick_grey)
        ax_top.grid(False)

        ax_bottom.set_xlim(ax.get_xlim())
        ax_bottom.xaxis.set_ticks_position('bottom')
        ax_bottom.spines['bottom'].set_position(('outward', 0))
        ax_bottom.set_xticks(top_ticks)
        ax_bottom.set_xticklabels([''] * len(top_ticks))
        ax_bottom.tick_params(axis='x', direction='out', length=2, colors=tick_grey, pad=2, top=False, bottom=True)
        ax_bottom.spines['bottom'].set_visible(False)
        ax_bottom.grid(False)

        y_low, y_high = ax.get_ylim()
        mirrored = [(y_low + y_high) - t for t in side_ticks]
        right_labels = list(reversed(side_labels))

        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_yticks(mirrored)
        ax_right.set_yticklabels(right_labels, fontsize=9, color=tick_grey)
        ax_right.tick_params(axis='y', direction='out', length=3, colors=tick_grey, pad=2, right=True, left=False)
        ax_right.spines['right'].set_visible(True)
        ax_right.spines['right'].set_linewidth(0.6)
        ax_right.spines['right'].set_color(tick_grey)
        ax_right.grid(False)

        ax_left.set_ylim(ax.get_ylim())
        ax_left.yaxis.set_ticks_position('left')
        ax_left.spines['left'].set_position(('outward', 0))
        ax_left.set_yticks(mirrored)
        ax_left.set_yticklabels([''] * len(mirrored))
        ax_left.tick_params(axis='y', direction='out', length=2, colors=tick_grey, pad=2, left=True, right=False)
        ax_left.spines['left'].set_visible(False)
        ax_left.grid(False)
    
    def _setup_mf34_heatmap_ticks(self, ax: plt.Axes, data: 'MF34HeatmapData') -> None:
        """
        Setup ticks for MF34 angular distribution heatmap.
        
        Implements complete 4-axis tick system with:
        - Log-scale decade ticks with scientific notation (1e+05 format)
        - Energy labels on TOP and RIGHT axes
        - Tick marks only on BOTTOM and LEFT axes
        - Special handling for inverted Y-axis (right labels reversed)
        - Support for multiple Legendre coefficient blocks
        """
        import numpy as np

        block_info = data.block_info or {}
        legendre_list = block_info.get('legendre_coeffs', data.legendre_coeffs)
        energy_grids = data.energy_grids or {}
        ranges_idx = block_info.get('ranges', {}) or {}
        energy_ranges = block_info.get('energy_ranges', {}) or {}
        edges_map = block_info.get('edges_transformed', {}) or {}
        tick_grey = "#707070"

        def _fmt_sci_eV(v: float) -> str:
            return f"{v:.0e}"

        def _log_ticks_with_decades(edges_block: np.ndarray):
            Emin = float(edges_block[0])
            Emax = float(edges_block[-1])
            kmin = int(np.ceil(np.log10(max(Emin, 1e-300))))
            kmax = int(np.floor(np.log10(max(Emax, 1e-300))))
            ks = np.arange(kmin, kmax + 1, dtype=int) if kmax >= kmin else np.array([kmin])

            vals: list[float] = []
            labels: list[str] = []

            is_emin_decade = np.isclose(Emin, 10.0 ** np.round(np.log10(Emin)), rtol=1e-10)
            is_emax_decade = np.isclose(Emax, 10.0 ** np.round(np.log10(Emax)), rtol=1e-10)
            if not is_emin_decade:
                vals.append(Emin)
                labels.append(_fmt_sci_eV(Emin))

            for k in ks:
                base = 10.0 ** k
                for m in range(1, 10):
                    v = m * base
                    if Emin <= v <= Emax:
                        vals.append(v)
                        labels.append(_fmt_sci_eV(v) if m == 1 else "")

            if not is_emax_decade and (len(vals) == 0 or not np.isclose(vals[-1], Emax, rtol=1e-10)):
                vals.append(Emax)
                labels.append(_fmt_sci_eV(Emax))

            vals_arr = np.array(vals, dtype=float)
            if vals_arr.size == 0:
                return np.array([]), []

            base0 = np.log10(max(Emin, 1e-300))
            return np.log10(vals_arr) - base0, labels

        def _linear_ticks_20pct(edges_block: np.ndarray):
            Emin = float(edges_block[0])
            Emax = float(edges_block[-1])
            if Emax <= Emin:
                return np.array([]), []
            fracs = np.linspace(0.0, 1.0, 6)
            vals = Emin + fracs * (Emax - Emin)
            labels = [_fmt_sci_eV(v) for v in vals]
            return vals - Emin, labels

        def _energy_ticks(edges_block: np.ndarray):
            if data.scale == "log" and edges_block[0] > 0:
                return _log_ticks_with_decades(edges_block)
            return _linear_ticks_20pct(edges_block)

        # Reconstruct transformed edges if they were not attached (fallback to concatenation order)
        if not edges_map:
            g_map = block_info.get("G_per_L", {})
            if data.is_diagonal and getattr(data, "x_edges", None) is not None:
                pointer = 0
                for l_val in legendre_list:
                    g_len = g_map.get(l_val)
                    if g_len:
                        edges_map[l_val] = np.asarray(data.x_edges[pointer:pointer + g_len + 1], dtype=float)
                        pointer += g_len
            elif not data.is_diagonal:
                if getattr(data, "x_edges", None) is not None and len(legendre_list) >= 2:
                    g_len = g_map.get(legendre_list[1])
                    if g_len:
                        edges_map[legendre_list[1]] = np.asarray(data.x_edges[:g_len + 1], dtype=float)
                if getattr(data, "y_edges", None) is not None and len(legendre_list) >= 1:
                    g_len = g_map.get(legendre_list[0])
                    if g_len:
                        edges_map[legendre_list[0]] = np.asarray(data.y_edges[:g_len + 1], dtype=float)

        # Primary-axis Legendre ticks
        if data.is_diagonal:
            centers = []
            labels = []
            for l_val in legendre_list:
                rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                if rng:
                    centers.append((rng[0] + rng[1]) * 0.5)
                    labels.append(str(l_val))
            if centers:
                ax.set_xticks(centers)
                ax.set_xticklabels(labels)
                ax.set_yticks(centers)
                ax.set_yticklabels(labels)
        elif len(legendre_list) == 2:
            row_l, col_l = legendre_list
            x_rng = energy_ranges.get(col_l) or ranges_idx.get(col_l)
            y_rng = energy_ranges.get(row_l) or ranges_idx.get(row_l)
            if x_rng:
                ax.set_xticks([(x_rng[0] + x_rng[1]) * 0.5])
                ax.set_xticklabels([str(col_l)])
            if y_rng:
                ax.set_yticks([(y_rng[0] + y_rng[1]) * 0.5])
                ax.set_yticklabels([str(row_l)])

        # Energy ticks might be disabled even if Legendre ticks are drawn
        if not data.show_energy_ticks or not energy_grids:
            return

        blocks_for_x = []
        blocks_for_y = []
        if data.is_diagonal:
            for l_val in legendre_list:
                raw_edges = energy_grids.get(l_val)
                transformed_edges = edges_map.get(l_val)
                if raw_edges is None or transformed_edges is None:
                    continue
                blocks_for_x.append((l_val, np.asarray(raw_edges, dtype=float), transformed_edges[0]))
            blocks_for_y = blocks_for_x
        elif len(legendre_list) == 2:
            row_l, col_l = legendre_list
            raw_y = energy_grids.get(row_l)
            raw_x = energy_grids.get(col_l)
            trans_y = edges_map.get(row_l)
            trans_x = edges_map.get(col_l)
            if raw_x is not None and trans_x is not None:
                blocks_for_x.append((col_l, np.asarray(raw_x, dtype=float), trans_x[0]))
            if raw_y is not None and trans_y is not None:
                blocks_for_y.append((row_l, np.asarray(raw_y, dtype=float), trans_y[0]))

        if not blocks_for_x or not blocks_for_y:
            return

        top_ticks: list[float] = []
        top_labels: list[str] = []
        side_ticks: list[float] = []
        side_labels: list[str] = []

        for i, (l_val, raw_edges, offset) in enumerate(blocks_for_x):
            pos_local, labels_all = _energy_ticks(raw_edges)
            pos_global = pos_local + offset

            filtered_top_labels = []
            for lbl in labels_all:
                if lbl and (lbl == labels_all[0] or lbl == labels_all[-1]):
                    filtered_top_labels.append("")
                else:
                    filtered_top_labels.append(lbl)
            top_ticks.extend(pos_global.tolist())
            top_labels.extend(filtered_top_labels)

            is_last_block = (i == len(blocks_for_x) - 1)
            side_ticks.extend(pos_global.tolist())
            filtered_side_labels = []
            for lbl in labels_all:
                if lbl and lbl == labels_all[-1]:
                    filtered_side_labels.append(lbl if is_last_block else "")
                else:
                    filtered_side_labels.append(lbl)
            side_labels.extend(filtered_side_labels)

        ax_top = ax.twiny()
        ax_right = ax.twinx()
        ax_bottom = ax.twiny()
        ax_left = ax.twinx()

        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(top_ticks)
        ax_top.set_xticklabels(top_labels, fontsize=9, color=tick_grey)
        ax_top.tick_params(axis='x', direction='out', length=3, colors=tick_grey, pad=2, top=True, bottom=False)
        ax_top.spines['top'].set_visible(True)
        ax_top.spines['top'].set_linewidth(0.6)
        ax_top.spines['top'].set_color(tick_grey)
        ax_top.grid(False)

        ax_bottom.set_xlim(ax.get_xlim())
        ax_bottom.xaxis.set_ticks_position('bottom')
        ax_bottom.spines['bottom'].set_position(('outward', 0))
        ax_bottom.set_xticks(top_ticks)
        ax_bottom.set_xticklabels([''] * len(top_ticks))
        ax_bottom.tick_params(axis='x', direction='out', length=2, colors=tick_grey, pad=2, top=False, bottom=True)
        ax_bottom.spines['bottom'].set_visible(False)
        ax_bottom.grid(False)

        y_low, y_high = (ax.get_ylim()[1], ax.get_ylim()[0]) if ax.get_ylim()[0] < ax.get_ylim()[1] else ax.get_ylim()
        if getattr(data, "y_edges", None) is not None:
            y_low, y_high = data.y_edges[0], data.y_edges[-1]
        mirrored = [(y_low + y_high) - t for t in side_ticks]
        right_labels = list(reversed(side_labels))

        ax_right.set_ylim(ax.get_ylim())
        ax_right.set_yticks(mirrored)
        ax_right.set_yticklabels(right_labels, fontsize=9, color=tick_grey)
        ax_right.tick_params(axis='y', direction='out', length=3, colors=tick_grey, pad=2, right=True, left=False)
        ax_right.spines['right'].set_visible(True)
        ax_right.spines['right'].set_linewidth(0.6)
        ax_right.spines['right'].set_color(tick_grey)
        ax_right.grid(False)

        ax_left.set_ylim(ax.get_ylim())
        ax_left.yaxis.set_ticks_position('left')
        ax_left.spines['left'].set_position(('outward', 0))
        ax_left.set_yticks(mirrored)
        ax_left.set_yticklabels([''] * len(mirrored))
        ax_left.tick_params(axis='y', direction='out', length=2, colors=tick_grey, pad=2, left=True, right=False)
        ax_left.spines['left'].set_visible(False)
        ax_left.grid(False)
    
    def _draw_block_boundaries(
        self,
        ax: plt.Axes,
        heatmap_data: 'HeatmapPlotData'
    ) -> None:
        """Draw light grid lines separating MT/L submatrices."""
        from .plot_data import CovarianceHeatmapData, MF34HeatmapData

        ranges_x = []
        ranges_y = []

        if isinstance(heatmap_data, CovarianceHeatmapData):
            info = heatmap_data.block_info or {}
            mts = info.get('mts', [])
            energy_ranges = info.get('energy_ranges', {}) or {}
            ranges_fallback = info.get('ranges', []) or []

            if heatmap_data.is_diagonal:
                for i, m in enumerate(mts):
                    rng = energy_ranges.get(m)
                    if rng is None and i < len(ranges_fallback):
                        rng = ranges_fallback[i]
                    if rng is not None:
                        ranges_x.append(tuple(rng))
                        ranges_y.append(tuple(rng))
            elif len(mts) >= 2:
                col_rng = energy_ranges.get(mts[1]) or (ranges_fallback[1] if len(ranges_fallback) > 1 else None)
                row_rng = energy_ranges.get(mts[0]) or (ranges_fallback[0] if ranges_fallback else None)
                if col_rng is not None:
                    ranges_x.append(tuple(col_rng))
                if row_rng is not None:
                    ranges_y.append(tuple(row_rng))

        elif isinstance(heatmap_data, MF34HeatmapData):
            info = heatmap_data.block_info or {}
            legendre_list = info.get('legendre_coeffs', heatmap_data.legendre_coeffs)
            energy_ranges = info.get('energy_ranges', {}) or {}
            ranges_idx = info.get('ranges', {}) or {}

            if heatmap_data.is_diagonal and len(legendre_list) > 1:
                for l_val in legendre_list:
                    rng = energy_ranges.get(l_val) or ranges_idx.get(l_val)
                    if rng is not None:
                        ranges_x.append(tuple(rng))
                        ranges_y.append(tuple(rng))
            elif len(legendre_list) >= 2:
                col_l = legendre_list[1]
                row_l = legendre_list[0]
                col_rng = energy_ranges.get(col_l) or ranges_idx.get(col_l)
                row_rng = energy_ranges.get(row_l) or ranges_idx.get(row_l)
                if col_rng is not None:
                    ranges_x.append(tuple(col_rng))
                if row_rng is not None:
                    ranges_y.append(tuple(row_rng))

        if len(ranges_x) <= 1 and len(ranges_y) <= 1:
            return

        def _boundary_positions(ranges):
            sorted_ranges = sorted(ranges, key=lambda r: r[0])
            return [r[0] for r in sorted_ranges[1:]]

        for x in _boundary_positions(ranges_x):
            ax.axvline(x, color="#404040", lw=1.0, alpha=0.8, zorder=5)
        for y in _boundary_positions(ranges_y):
            ax.axhline(y, color="#404040", lw=1.0, alpha=0.8, zorder=5)
    
    def _draw_uncertainty_panels(
        self,
        uncertainty_axes: List[plt.Axes],
        heatmap_data: 'HeatmapPlotData',
        ax_heatmap: plt.Axes
    ) -> None:
        """
        Draw uncertainty panels above heatmap.
        
        Uses step plotting with 'post' stepping for proper bin coverage,
        adds internal grid lines at 10% intervals with right-aligned labels.
        """
        from .plot_data import CovarianceHeatmapData, MF34HeatmapData
        import numpy as np
        
        uncertainty_data = heatmap_data.uncertainty_data
        if not uncertainty_data:
            return
        
        background_color = "#F5F5F5"
        tick_grey = "#707070"
        
        # Helper function to draw internal y-grid
        def _nice_ylim_and_step(max_val: float) -> tuple[float, float]:
            """Return (top, step) for readable grids with ~5-8 ticks."""
            if not np.isfinite(max_val) or max_val <= 0:
                return 1.0, 0.2
            base = 10.0 ** np.floor(np.log10(max_val))
            for mult in (1.0, 2.0, 5.0, 10.0):
                step = mult * base
                if max_val / step <= 8.0:
                    break
            top = np.ceil(max_val / step) * step
            return float(top), float(step)

        def _draw_ygrid_inside(ax_u, xr, ymax):
            """Draw grid lines with adaptive spacing and right-aligned labels inside uncertainty panel."""
            top, step = _nice_ylim_and_step(ymax)
            grid_vals = np.arange(0.0, top + 1e-9, step)
            for y in grid_vals:
                ax_u.axhline(y, color=tick_grey, lw=0.6, alpha=0.35, zorder=0)
            x_label = xr[1] - 0.01 * (xr[1] - xr[0])
            for y in grid_vals:
                lbl = f"{y:g}" if step < 1 else f"{int(y)}"
                ax_u.text(x_label, y, lbl, ha="right", va="center",
                         color=tick_grey, fontsize=8, alpha=0.9, zorder=2)
        
        # Get sorted keys (MTs or Legendre orders)
        keys = sorted(uncertainty_data.keys())
        
        if isinstance(heatmap_data, MF34HeatmapData):
            # MF34 uncertainty panels
            # Use transformed energy coordinates matching the heatmap
            edges_map = {}
            if heatmap_data.block_info:
                edges_map = heatmap_data.block_info.get("edges_transformed", {}) or {}

            def _edges_for_L(L_val: int) -> Optional[np.ndarray]:
                if L_val in edges_map:
                    return np.asarray(edges_map[L_val], dtype=float)
                raw = heatmap_data.energy_grids.get(L_val)
                if raw is None:
                    return None
                if heatmap_data.scale == "log":
                    transformed = np.log10(np.maximum(raw, 1e-300))
                else:
                    transformed = np.asarray(raw, dtype=float)
                return transformed - transformed[0]

            if heatmap_data.is_diagonal and len(keys) > 1:
                # Multiple diagonal blocks - each panel shows one L using global coords
                for i, (ax_u, L) in enumerate(zip(uncertainty_axes, keys)):
                    sigma_pct = uncertainty_data[L]
                    xs_edges = _edges_for_L(L)
                    if xs_edges is None or xs_edges.size == 0:
                        continue

                    ax_u.set_facecolor(background_color)
                    ax_u.grid(False)

                    # Use local coordinates so each panel width matches its own block
                    xs_local = xs_edges - xs_edges[0]
                    sigma_vals = sigma_pct if sigma_pct.size == xs_local.size - 1 else sigma_pct
                    if sigma_vals.size > 0:
                        sigma_ext = np.append(sigma_vals, sigma_vals[-1])
                    else:
                        sigma_ext = sigma_vals

                    if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                        ax_u.step(xs_local, sigma_ext, where='post',
                                  linewidth=1.4, color=f"C{i}", zorder=3)
                        y_max = float(np.nanmax(sigma_ext))
                    else:
                        y_max = 5.0

                    y_max = max(y_max, 1.0)
                    top, _ = _nice_ylim_and_step(y_max * 1.05)
                    ax_u.set_ylim(0.0, top)
                    ax_u.set_xlim(0.0, xs_local[-1])

                    ax_u.set_xticks([])
                    ax_u.set_yticks([])

                    _draw_ygrid_inside(ax_u, (0.0, xs_local[-1]), ax_u.get_ylim()[1])

                    if i == 0:
                        ax_u.set_ylabel('Unc. (%)', fontsize=10, color='black')

                    for side in ('left', 'right', 'top', 'bottom'):
                        ax_u.spines[side].set_visible(False)
            else:
                # Single block (one L or off-diagonal)
                ax_u = uncertainty_axes[0]
                L = keys[0]
                sigma_pct = uncertainty_data[L]
                xs_edges = _edges_for_L(L)
                if xs_edges is None or xs_edges.size == 0:
                    return

                ax_u.set_facecolor(background_color)
                ax_u.grid(False)

                xs_local = xs_edges - xs_edges[0]
                sigma_ext = np.append(sigma_pct, sigma_pct[-1]) if sigma_pct.size > 0 else sigma_pct

                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    ax_u.step(xs_local, sigma_ext, where='post',
                              linewidth=1.4, color='C0', zorder=3)
                    y_max = float(np.nanmax(sigma_ext))
                else:
                    y_max = 5.0

                y_max = max(y_max, 1.0)
                top, _ = _nice_ylim_and_step(y_max * 1.05)
                ax_u.set_ylim(0.0, top)

                ax_u.set_xlim(0.0, xs_local[-1])
                ax_u.set_xticks([])
                ax_u.set_yticks([])

                _draw_ygrid_inside(ax_u, (0.0, xs_local[-1]), ax_u.get_ylim()[1])

                ax_u.set_ylabel('Unc. (%)', fontsize=10)

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_u.spines[side].set_visible(False)
        
        elif isinstance(heatmap_data, CovarianceHeatmapData):
            # CovMat uncertainty panels
            energy_grid = heatmap_data.energy_grid

            if energy_grid is not None:
                if heatmap_data.scale == "log":
                    edges_tx = np.log10(np.maximum(energy_grid, 1e-300))
                else:
                    edges_tx = np.asarray(energy_grid, dtype=float)
                edges_tx = edges_tx - edges_tx[0]
            else:
                edges_tx = None
            
            for i, (ax_unc, key) in enumerate(zip(uncertainty_axes, keys)):
                sigma_pct = uncertainty_data[key]

                if edges_tx is not None and len(edges_tx) == len(sigma_pct) + 1:
                    xs_edges = edges_tx
                else:
                    xs_edges = np.arange(len(sigma_pct) + 1, dtype=float)

                sigma_ext = np.append(sigma_pct, sigma_pct[-1]) if sigma_pct.size > 0 else sigma_pct
                xs_plot = xs_edges

                ax_unc.set_facecolor(background_color)
                ax_unc.grid(False)

                color = self._colors[i % len(self._colors)] if getattr(self, "_colors", None) else f"C{i}"
                if sigma_ext.size > 0 and np.any(sigma_ext > 0):
                    ax_unc.step(xs_plot, sigma_ext, where='post',
                                linewidth=1.4, color=color, zorder=3)
                    y_max = float(np.nanmax(sigma_ext))
                else:
                    y_max = 5.0

                y_max = max(y_max, 1.0)
                top, _ = _nice_ylim_and_step(y_max * 1.05)
                ax_unc.set_ylim(0.0, top)
                ax_unc.set_xlim(xs_plot[0], xs_plot[-1])

                ax_unc.set_xticks([])
                ax_unc.set_yticks([])

                if xs_edges.size >= 2:
                    _draw_ygrid_inside(ax_unc, (xs_edges[0], xs_edges[-1]), ax_unc.get_ylim()[1])

                if i == 0:
                    ax_unc.set_ylabel('Unc. (%)', fontsize=10, color='black')

                for side in ('left', 'right', 'top', 'bottom'):
                    ax_unc.spines[side].set_visible(False)
    
    def build(self, show: bool = False) -> plt.Figure:
        """
        Build and return the figure.
        
        Parameters
        ----------
        show : bool
            Whether to display the figure immediately
            
        Returns
        -------
        matplotlib.figure.Figure
            The completed figure
        """
        # Check if we're building a heatmap or line plot
        if self._heatmap_data is not None:
            # Build heatmap
            fig = self._build_heatmap()
            if show:
                plt.show()
            return fig
        
        # Otherwise, build line plot (existing logic)
        # Get default colors
        default_colors = self._colors
        
        # Plot uncertainty bands first (so they appear behind the lines)
        for band, data_idx in self._uncertainty_bands:
            # Use the same color as the associated data if not specified
            fill_kwargs = band.get_fill_kwargs()
            
            if band.color is None and data_idx < len(self._data_list):
                # Get color from associated data or custom styling
                if 'color' in self._custom_styling[data_idx]:
                    fill_kwargs['color'] = self._custom_styling[data_idx]['color']
                elif self._data_list[data_idx].color is not None:
                    fill_kwargs['color'] = self._data_list[data_idx].color
            
            # Check if the associated data is a step plot (for multigroup/uncertainty data)
            data = self._data_list[data_idx] if data_idx < len(self._data_list) else None
            
            # Convert relative uncertainties to absolute if needed
            if band.is_relative():
                if data is None:
                    raise ValueError("Cannot convert relative uncertainty to absolute without associated data")
                y_lower, y_upper = band.to_absolute(data.y)
            else:
                y_lower, y_upper = band.y_lower, band.y_upper
            
            if data and data.plot_type == 'step':
                # For step plots, use step='post' to make uncertainty bands follow the steps
                self.ax.fill_between(band.x, y_lower, y_upper, step='post', **fill_kwargs)
            else:
                # Regular smooth fill for line plots
                self.ax.fill_between(band.x, y_lower, y_upper, **fill_kwargs)
        
        # Plot each data object
        for i, (data, styling_overrides) in enumerate(zip(self._data_list, self._custom_styling)):
            # Merge styling: data defaults < custom overrides
            plot_kwargs = data.get_plot_kwargs()
            plot_kwargs.update(styling_overrides)
            
            # Auto-assign color if not specified
            if 'color' not in plot_kwargs and default_colors is not None:
                plot_kwargs['color'] = default_colors[i % len(default_colors)]
            
            # Plot based on plot_type
            if data.plot_type == 'line':
                self.ax.plot(data.x, data.y, **plot_kwargs)
            
            elif data.plot_type == 'step':
                # Handle step plots (common for uncertainties and multigroup data)
                # Use 'post' where parameter to match the old plotting method
                where = getattr(data, 'step_where', 'post')
                
                # Check if markers are requested
                marker = plot_kwargs.get('marker', None)
                if marker is not None and marker != '':
                    # For step plots with markers, we need to:
                    # 1. Plot the step line without markers
                    # 2. Plot markers separately at segment midpoints
                    
                    # Extract marker properties
                    markersize = plot_kwargs.pop('markersize', None)
                    markerfacecolor = plot_kwargs.get('color', None)
                    markeredgecolor = plot_kwargs.get('color', None)
                    
                    # Plot step line without marker
                    marker_backup = plot_kwargs.pop('marker')
                    self.ax.step(data.x, data.y, where=where, **plot_kwargs)
                    
                    # Calculate midpoints for markers
                    # For 'post' step: each segment spans from x[i] to x[i+1] at height y[i]
                    # For 'pre' step: each segment spans from x[i-1] to x[i] at height y[i]
                    # For 'mid' step: segments are centered at x[i]
                    
                    if where == 'post' and len(data.x) > 1:
                        # Midpoints between consecutive x values
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[:-1]  # Heights correspond to the left point
                    elif where == 'pre' and len(data.x) > 1:
                        x_mid = (data.x[:-1] + data.x[1:]) / 2
                        y_mid = data.y[1:]  # Heights correspond to the right point
                    elif where == 'mid':
                        x_mid = data.x
                        y_mid = data.y
                    else:
                        # Fallback: use original points
                        x_mid = data.x
                        y_mid = data.y
                    
                    # Plot markers at midpoints
                    marker_kwargs = {
                        'marker': marker_backup,
                        'linestyle': 'none',
                        'color': markerfacecolor,
                        'markeredgecolor': markeredgecolor,
                    }
                    if markersize is not None:
                        marker_kwargs['markersize'] = markersize
                    if 'alpha' in plot_kwargs:
                        marker_kwargs['alpha'] = plot_kwargs['alpha']
                    
                    self.ax.plot(x_mid, y_mid, **marker_kwargs)
                else:
                    # No markers, just plot the step normally
                    self.ax.step(data.x, data.y, where=where, **plot_kwargs)
            
            elif data.plot_type == 'scatter':
                # Convert markersize to s for scatter plots
                if 'markersize' in plot_kwargs:
                    plot_kwargs['s'] = plot_kwargs.pop('markersize')
                self.ax.scatter(data.x, data.y, **plot_kwargs)
            
            elif data.plot_type == 'errorbar':
                # Extract error bar specific kwargs
                yerr = plot_kwargs.pop('yerr', None)
                xerr = plot_kwargs.pop('xerr', None)
                self.ax.errorbar(data.x, data.y, yerr=yerr, xerr=xerr, **plot_kwargs)
            
            else:
                raise ValueError(f"Unknown plot_type: {data.plot_type}")
        
        # Apply scales
        if self._use_log_x:
            self.ax.set_xscale('log')
        if self._use_log_y:
            self.ax.set_yscale('log')
        
        # Set tight limits by default (no margins) if limits are not specified
        if self._x_lim is None and self._data_list:
            # Find the data range across all datasets
            x_min = min(np.min(data.x) for data in self._data_list if len(data.x) > 0)
            x_max = max(np.max(data.x) for data in self._data_list if len(data.x) > 0)
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                self.ax.set_xlim(x_min, x_max)
        else:
            # Apply user-specified limits
            if self._x_lim is not None:
                self.ax.set_xlim(self._x_lim)
        
        if self._y_lim is None and self._data_list:
            # Find the data range across all datasets (including uncertainty bands)
            y_values = []
            for data in self._data_list:
                if len(data.y) > 0:
                    y_values.extend(data.y)
            # Also include uncertainty band values
            for band, data_idx in self._uncertainty_bands:
                if band.is_relative():
                    # Need to convert to absolute first
                    if data_idx < len(self._data_list):
                        data = self._data_list[data_idx]
                        y_lower, y_upper = band.to_absolute(data.y)
                        if len(y_lower) > 0:
                            y_values.extend(y_lower)
                        if len(y_upper) > 0:
                            y_values.extend(y_upper)
                else:
                    # Already absolute
                    if band.y_lower is not None and len(band.y_lower) > 0:
                        y_values.extend(band.y_lower)
                    if band.y_upper is not None and len(band.y_upper) > 0:
                        y_values.extend(band.y_upper)
            
            if y_values:
                y_arr = np.array(y_values)
                y_arr = y_arr[np.isfinite(y_arr)]  # Remove inf/nan
                if len(y_arr) > 0:
                    y_min, y_max = np.min(y_arr), np.max(y_arr)
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                        # Add small padding for y-axis (5% on each side)
                        if self._use_log_y and y_min > 0:
                            # For log scale, use multiplicative padding
                            log_range = np.log10(y_max) - np.log10(y_min)
                            padding = 0.05 * log_range
                            y_min = 10 ** (np.log10(y_min) - padding)
                            y_max = 10 ** (np.log10(y_max) + padding)
                        else:
                            # For linear scale, use additive padding
                            padding = 0.05 * (y_max - y_min)
                            y_min = y_min - padding
                            y_max = y_max + padding
                        self.ax.set_ylim(y_min, y_max)
        else:
            # Apply user-specified limits
            if self._y_lim is not None:
                self.ax.set_ylim(self._y_lim)
        
        # Apply axis labels
        if self._x_label is not None:
            if self._label_fontsize is not None:
                self.ax.set_xlabel(self._x_label, fontsize=self._label_fontsize)
            else:
                self.ax.set_xlabel(self._x_label)
        elif self._use_log_x:
            # Auto-label for energy axis if log scale is used
            is_energy_axis = self._x_label is None or 'energy' in self._x_label.lower()
            if is_energy_axis:
                self.ax.set_xlabel("Energy (MeV)")
        else:
            # Auto-label for group index if not log scale and no label
            if self._x_label is None:
                self.ax.set_xlabel("Energy-group index")
                self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        
        if self._y_label is not None:
            if self._label_fontsize is not None:
                self.ax.set_ylabel(self._y_label, fontsize=self._label_fontsize)
            else:
                self.ax.set_ylabel(self._y_label)
        
        # Apply title
        if self._title is not None:
            if self._title_fontsize is not None:
                self.ax.set_title(self._title, fontsize=self._title_fontsize)
            else:
                self.ax.set_title(self._title)
        
        # Format energy axis ticks if using log scale on x-axis
        if self._use_log_x:
            format_energy_axis_ticks(self.ax)
        
        # Apply tick label size
        if self._tick_labelsize is not None:
            self.ax.tick_params(axis='both', which='major', labelsize=self._tick_labelsize)
        
        # Add legend if there are labeled artists
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            legend_kwargs = {'loc': self._legend_loc, 'framealpha': 0.9}
            
            if self.style == 'light':
                # For light style, use framed legend with black edge (publication style)
                legend_kwargs.update({
                    'frameon': True,
                    'fancybox': False,
                    'edgecolor': 'black'
                })
            else:
                # For dark style, use default fancybox
                legend_kwargs['fancybox'] = True
            
            if self._legend_fontsize is not None:
                legend_kwargs['fontsize'] = self._legend_fontsize
            
            self.ax.legend(**legend_kwargs)
        
        # Apply tick parameters if specified
        if hasattr(self, '_tick_params') and self._tick_params:
            from matplotlib.ticker import MaxNLocator
            
            # Limit number of ticks
            if 'max_ticks_x' in self._tick_params:
                max_x = self._tick_params['max_ticks_x']
                if self._use_log_x:
                    # For log scale, use LogLocator with numticks parameter
                    from matplotlib.ticker import LogLocator
                    self.ax.xaxis.set_major_locator(LogLocator(numticks=max_x))
                else:
                    # For linear scale, use MaxNLocator
                    self.ax.xaxis.set_major_locator(MaxNLocator(nbins=max_x, integer=False))
            
            if 'max_ticks_y' in self._tick_params:
                max_y = self._tick_params['max_ticks_y']
                if self._use_log_y:
                    from matplotlib.ticker import LogLocator
                    self.ax.yaxis.set_major_locator(LogLocator(numticks=max_y))
                else:
                    self.ax.yaxis.set_major_locator(MaxNLocator(nbins=max_y, integer=False))
            
            # Rotate tick labels
            if 'rotate_x' in self._tick_params:
                rotation = self._tick_params['rotate_x']
                # Also adjust horizontal alignment for better appearance
                ha = 'right' if rotation > 0 else 'center'
                self.ax.tick_params(axis='x', rotation=rotation)
                for label in self.ax.get_xticklabels():
                    label.set_rotation(rotation)
                    label.set_ha(ha)
            
            if 'rotate_y' in self._tick_params:
                rotation = self._tick_params['rotate_y']
                self.ax.tick_params(axis='y', rotation=rotation)
                for label in self.ax.get_yticklabels():
                    label.set_rotation(rotation)
        
        # Apply tight layout to prevent label cutoff (especially after rotation)
        # Only if constrained_layout is not already being used
        if hasattr(self, '_tick_params') and self._tick_params:
            try:
                # Check if constrained_layout is active
                if not self.fig.get_constrained_layout():
                    self.fig.tight_layout()
            except:
                # Fallback if constrained_layout check fails
                try:
                    self.fig.tight_layout()
                except:
                    pass  # If tight_layout fails, just continue
        
        # Apply grid configuration
        if self._grid:
            # Major grid
            self.ax.grid(True, which='major', linestyle='--', alpha=self._grid_alpha)
            
            # Minor grid (enabled for light style by default, or if explicitly requested)
            if self._show_minor_grid or (self.style == 'light' and not hasattr(self, '_grid_configured')):
                self.ax.minorticks_on()
                minor_alpha = self._minor_grid_alpha if self._show_minor_grid else 0.25
                self.ax.grid(True, which='minor', linestyle=':', alpha=minor_alpha, linewidth=0.5)
        else:
            self.ax.grid(False)
        
        # Finalize and display
        if show:
            if self._notebook_mode:
                # In notebooks, the plot should auto-display
                # Only call show() if using inline backend
                if plt.get_backend() == 'module://matplotlib_inline.backend_inline':
                    plt.show()
            else:
                plt.show()
        
        return self.fig
    
    def _convert_plotdata_to_band(
        self,
        uncertainty_data: PlotData,
        nominal_data: PlotData
    ) -> UncertaintyBand:
        """
        Convert uncertainty PlotData to UncertaintyBand.
        
        Parameters
        ----------
        uncertainty_data : PlotData
            PlotData containing uncertainty values (MultigroupUncertaintyPlotData or LegendreUncertaintyPlotData)
        nominal_data : PlotData
            The nominal data that these uncertainties correspond to
            
        Returns
        -------
        UncertaintyBand
            Converted uncertainty band object
        """
        # Check if this is a recognized uncertainty PlotData type
        if isinstance(uncertainty_data, (MultigroupUncertaintyPlotData, LegendreUncertaintyPlotData)):
            # These PlotData types now store uncertainties at bin edges (n+1 points)
            # to properly support step plots with where='post'
            
            uncertainty_type = getattr(uncertainty_data, 'uncertainty_type', 'relative')
            
            if uncertainty_type == 'relative':
                # y values are in percentage - convert to fractional (0.05 for 5%)
                rel_unc = np.array(uncertainty_data.y) / 100.0
                unc_x = np.array(uncertainty_data.x)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate uncertainty to match nominal grid
                    # This happens when ENDF MF34 (sparse) is combined with MF4 (dense)
                    rel_unc = np.interp(nom_x, unc_x, rel_unc)
                elif len(nominal_data.y) == len(rel_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case - G vs G+1 points)
                    # Extend uncertainties to match nominal data
                    rel_unc = np.append(rel_unc, rel_unc[-1])
                
                # Use nominal data's x values (energy bin boundaries)
                return UncertaintyBand(
                    x=nominal_data.x,
                    relative_uncertainty=rel_unc,
                    sigma=1.0,  # Uncertainties are already at the desired sigma level
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
            else:  # absolute
                # y values are absolute uncertainties
                abs_unc = np.array(uncertainty_data.y)
                unc_x = np.array(uncertainty_data.x)
                nominal_y = np.array(nominal_data.y)
                nom_x = np.array(nominal_data.x)
                
                # Check if grids match
                if len(unc_x) != len(nom_x) or not np.allclose(unc_x, nom_x):
                    # Different grids - need to interpolate
                    abs_unc = np.interp(nom_x, unc_x, abs_unc)
                elif len(nominal_y) == len(abs_unc) + 1:
                    # Same grid but uncertainty is shorter (legacy case)
                    abs_unc = np.append(abs_unc, abs_unc[-1])
                
                # Convert to bounds
                y_lower = nominal_y - abs_unc
                y_upper = nominal_y + abs_unc
                
                return UncertaintyBand(
                    x=nominal_data.x,
                    y_lower=y_lower,
                    y_upper=y_upper,
                    color=getattr(uncertainty_data, 'color', None),
                    alpha=0.2
                )
        else:
            # For other PlotData types, assume y values are relative uncertainties in fractional form
            rel_unc = np.array(uncertainty_data.y)
            
            # Check if we need to extend
            if len(nominal_data.y) == len(rel_unc) + 1:
                rel_unc = np.append(rel_unc, rel_unc[-1])
            
            return UncertaintyBand(
                x=nominal_data.x,
                relative_uncertainty=rel_unc,
                sigma=1.0,
                color=getattr(uncertainty_data, 'color', None),
                alpha=0.2
            )
    
    def clear(self) -> 'PlotBuilder':
        """
        Clear all data and reset to initial state.
        
        Returns
        -------
        PlotBuilder
            Self for method chaining
        """
        self._data_list.clear()
        self._uncertainty_bands.clear()
        self._custom_styling.clear()
        
        if self.ax is not None:
            self.ax.clear()
        
        return self
