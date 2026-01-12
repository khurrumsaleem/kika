"""
Sensitivity Analysis and Uncertainty Quantification Report Generators.

This module provides standardized report generators for sensitivity analysis results
from different sources (MCNP PERT, Serpent) and uncertainty quantification using
the sandwich formula.

Report Classes:
- MCNPPertReport: Full report for MCNP PERT sensitivity analysis
- SerpentSensReport: Full report for Serpent sensitivity analysis  
- ComparisonReport: Compare sensitivities from different sources via SDF
- UQReport: Uncertainty quantification report using sandwich formula

All reports can generate:
- Summary tables
- Sensitivity profile plots
- Interactive HTML or static figures
- SDFData for downstream processing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
import warnings
import io
import base64

from kika.sensitivities.sensitivity import SensitivityData, TaylorCoefficients
from kika.sensitivities.sdf import SDFData, SDFReactionData
from kika.sensitivities.sensitivity_processing import create_sdf_data
from kika._constants import MT_TO_REACTION, ATOMIC_NUMBER_TO_SYMBOL


# ============================================================================
# Utility Functions
# ============================================================================

def _format_nuclide(zaid: int) -> str:
    """Format ZAID as human-readable nuclide string."""
    z = zaid // 1000
    a = zaid % 1000
    if z in ATOMIC_NUMBER_TO_SYMBOL:
        return f"{ATOMIC_NUMBER_TO_SYMBOL[z]}-{a}"
    return f"Z{z}-{a}"


def _format_reaction(mt: int) -> str:
    """Format MT number as reaction name."""
    if mt in MT_TO_REACTION:
        return MT_TO_REACTION[mt]
    return f"MT{mt}"


def _fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _generate_html_header(title: str) -> str:
    """Generate HTML header with styling."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .report-container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f7ff;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            color: white;
            margin-top: 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .figure-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #0c5460;
            color: #0c5460;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
"""


def _generate_html_footer() -> str:
    """Generate HTML footer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
    <div class="footer">
        <p>Report generated by KIKA Sensitivity Analysis Module</p>
        <p>Generated on: {timestamp}</p>
    </div>
</body>
</html>
"""


# ============================================================================
# MCNP PERT Report
# ============================================================================

@dataclass
class MCNPPertReport:
    """
    Generate comprehensive sensitivity analysis reports from MCNP PERT results.
    
    This report class takes SensitivityData objects from MCNP PERT card processing
    and generates:
    1. Sensitivity coefficient tables and plots
    2. Second-order analysis (if available): ratios, perturbed response comparisons
    3. SDFData export for downstream UQ
    
    Parameters
    ----------
    sensitivity_data : SensitivityData or List[SensitivityData]
        Sensitivity data from MCNP PERT card processing
    title : str, optional
        Report title
        
    Attributes
    ----------
    data : List[SensitivityData]
        List of sensitivity data objects
    title : str
        Report title
    has_second_order : bool
        Whether second-order coefficients are available
    """
    
    sensitivity_data: Union[SensitivityData, List[SensitivityData]]
    title: str = "MCNP PERT Sensitivity Analysis Report"
    
    # Computed attributes
    data: List[SensitivityData] = field(init=False)
    has_second_order: bool = field(init=False)
    
    def __post_init__(self):
        """Initialize and validate data."""
        # Normalize to list
        if isinstance(self.sensitivity_data, SensitivityData):
            self.data = [self.sensitivity_data]
        else:
            self.data = list(self.sensitivity_data)
        
        # Check for second-order data
        self.has_second_order = self._check_second_order()
    
    def _check_second_order(self) -> bool:
        """Check if any sensitivity data contains second-order coefficients."""
        for sd in self.data:
            if sd.coefficients:
                for energy_data in sd.coefficients.values():
                    for coeff_obj in energy_data.values():
                        if hasattr(coeff_obj, 'c2') and any(
                            c2 != 0 for c2 in coeff_obj.c2 if not np.isnan(c2)
                        ):
                            return True
        return False
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of all sensitivity data.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: nuclide, tally, n_reactions, n_energies, has_2nd_order
        """
        rows = []
        for sd in self.data:
            has_2nd = False
            if sd.coefficients:
                for energy_data in sd.coefficients.values():
                    for coeff_obj in energy_data.values():
                        if hasattr(coeff_obj, 'c2') and any(
                            c2 != 0 for c2 in coeff_obj.c2 if not np.isnan(c2)
                        ):
                            has_2nd = True
                            break
            
            rows.append({
                'Nuclide': sd.nuclide,
                'ZAID': sd.zaid,
                'Tally ID': sd.tally_id,
                'Tally Name': sd.tally_name or 'N/A',
                'Label': sd.label,
                'Reactions': len(sd.reactions),
                'Energy Bins': len(sd.energies),
                'Pert. Bins': len(sd.pert_energies) - 1,
                'Second Order': 'Yes' if has_2nd else 'No'
            })
        
        return pd.DataFrame(rows)
    
    def sensitivity_table(
        self, 
        energy: str = 'integral',
        reactions: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate sensitivity coefficient table for specified energy and reactions.
        
        Parameters
        ----------
        energy : str
            Energy bin key ('integral' or 'lower_upper' format)
        reactions : List[int], optional
            MT numbers to include. If None, includes all.
            
        Returns
        -------
        pd.DataFrame
            Table with integrated sensitivities per nuclide/reaction
        """
        rows = []
        for sd in self.data:
            if energy not in sd.data:
                continue
            
            energy_data = sd.data[energy]
            for mt, coeff in energy_data.items():
                if reactions is not None and mt not in reactions:
                    continue
                
                # Calculate integrated sensitivity
                int_sens = sum(coeff.values) if hasattr(coeff, 'values') else 0
                # Note: coeff.errors contains RELATIVE errors from MCTAL (e.g., 0.05 for 5%)
                # These are combined in quadrature from detector and perturbation uncertainties
                int_err_rel = np.sqrt(sum(e**2 for e in coeff.errors)) if hasattr(coeff, 'errors') else 0
                # Calculate absolute uncertainty for error bars in plots
                int_err_abs = int_err_rel * abs(int_sens) if int_sens != 0 else 0
                
                rows.append({
                    'Nuclide': sd.nuclide,
                    'ZAID': sd.zaid,
                    'Reaction': _format_reaction(mt),
                    'MT': mt,
                    'Integrated Sensitivity': int_sens,
                    'Abs. Uncertainty': int_err_abs,
                    'Rel. Error (%)': abs(int_err_rel * 100)  # Convert from fraction to percentage
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Integrated Sensitivity', key=abs, ascending=False)
        return df
    
    def plot_sensitivities(
        self,
        energy: str = 'integral',
        reactions: Optional[List[int]] = None,
        top_n: Optional[int] = 10,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """
        Plot sensitivity profiles.
        
        Parameters
        ----------
        energy : str
            Energy bin key
        reactions : List[int], optional
            MT numbers to include
        top_n : int, optional
            Number of top sensitivities to plot
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
            Matplotlib figure
        """
        table = self.sensitivity_table(energy=energy, reactions=reactions)
        if table.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No sensitivity data available', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Take top N by absolute value
        if top_n and len(table) > top_n:
            table = table.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [f"{row['Nuclide']} {row['Reaction']}" for _, row in table.iterrows()]
        values = table['Integrated Sensitivity'].values
        errors = table['Abs. Uncertainty'].values
        
        colors = ['#3498db' if v >= 0 else '#e74c3c' for v in values]
        
        bars = ax.barh(range(len(values)), values, xerr=errors, 
                      color=colors, alpha=0.8, capsize=3)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Integrated Sensitivity')
        ax.set_title(f'Top {len(table)} Sensitivity Coefficients ({energy})')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_second_order_analysis(
        self,
        energy: str = 'integral',
        reaction: Optional[int] = None,
        nuclide_idx: int = 0
    ) -> Optional[Figure]:
        """
        Plot second-order analysis (ratio plots) if available.
        
        Parameters
        ----------
        energy : str
            Energy bin key
        reaction : int, optional
            Specific MT number. If None, plots first available.
        nuclide_idx : int
            Index of nuclide in data list
            
        Returns
        -------
        Figure or None
            Matplotlib figure, or None if no second-order data
        """
        if not self.has_second_order:
            return None
        
        if nuclide_idx >= len(self.data):
            return None
        
        sd = self.data[nuclide_idx]
        
        if energy not in sd.coefficients:
            return None
        
        coeff_data = sd.coefficients[energy]
        
        if reaction is None:
            # Get first available reaction
            if not coeff_data:
                return None
            reaction = list(coeff_data.keys())[0]
        
        if reaction not in coeff_data:
            return None
        
        taylor = coeff_data[reaction]
        
        # Use the built-in plot method
        fig, ax = plt.subplots(figsize=(10, 5))
        taylor.plot(ax=ax, title=f"{sd.nuclide} - {_format_reaction(reaction)} ({energy})")
        
        return fig
    
    def to_sdf(
        self,
        energy: str = 'integral',
        title: Optional[str] = None,
        reactions: Optional[Dict[int, List[int]]] = None
    ) -> SDFData:
        """
        Convert sensitivity data to SDF format.
        
        Parameters
        ----------
        energy : str
            Energy bin key to export
        title : str, optional
            SDF title. If None, uses report title.
        reactions : Dict[int, List[int]], optional
            ZAID -> MT list mapping to filter reactions
            
        Returns
        -------
        SDFData
            SDF formatted sensitivity data
        """
        if reactions is not None:
            # Convert to format expected by create_sdf_data
            sens_list = [(sd, reactions.get(sd.zaid, sd.reactions)) for sd in self.data]
        else:
            sens_list = self.data
        
        return create_sdf_data(
            sens_list=sens_list,
            energy=energy,
            title=title or self.title.replace(' ', '_')
        )
    
    def generate_html(
        self,
        energy: str = 'integral',
        include_plots: bool = True,
        include_second_order: bool = True
    ) -> str:
        """
        Generate complete HTML report.
        
        Parameters
        ----------
        energy : str
            Energy bin for main tables/plots
        include_plots : bool
            Whether to include embedded plots
        include_second_order : bool
            Whether to include second-order analysis (if available)
            
        Returns
        -------
        str
            Complete HTML document
        """
        html = _generate_html_header(self.title)
        
        html += '<div class="report-container">'
        html += f'<h1>{self.title}</h1>'
        
        # Summary section
        html += '<h2>Summary</h2>'
        summary_df = self.summary()
        html += summary_df.to_html(index=False, classes='summary-table')
        
        # Summary metrics
        html += '<div class="summary-box">'
        html += '<h3>Key Metrics</h3>'
        html += f'<div class="metric"><span class="metric-value">{len(self.data)}</span>'
        html += '<span class="metric-label"> Nuclides</span></div>'
        
        total_reactions = sum(len(sd.reactions) for sd in self.data)
        html += f'<div class="metric"><span class="metric-value">{total_reactions}</span>'
        html += '<span class="metric-label"> Total Reactions</span></div>'
        
        html += f'<div class="metric"><span class="metric-value">{"Yes" if self.has_second_order else "No"}</span>'
        html += '<span class="metric-label"> Second-Order Data</span></div>'
        html += '</div>'
        
        # Sensitivity table
        html += '<h2>Sensitivity Coefficients</h2>'
        sens_table = self.sensitivity_table(energy=energy)
        if not sens_table.empty:
            html += sens_table.to_html(index=False, float_format='%.4e')
        else:
            html += '<p class="warning">No sensitivity data for selected energy bin.</p>'
        
        # Plots
        if include_plots and not sens_table.empty:
            html += '<h2>Sensitivity Profile</h2>'
            html += '<div class="figure-container">'
            fig = self.plot_sensitivities(energy=energy)
            html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Sensitivity Plot">'
            plt.close(fig)
            html += '</div>'
        
        # Second-order analysis
        if include_second_order and self.has_second_order:
            html += '<h2>Second-Order Analysis</h2>'
            html += '<div class="info">Second-order Taylor coefficients are available. '
            html += 'These show the nonlinearity of the response with respect to cross-section perturbations.</div>'
            
            # Plot for each nuclide
            for i, sd in enumerate(self.data):
                if sd.coefficients and energy in sd.coefficients:
                    html += f'<h3>{sd.nuclide}</h3>'
                    fig = self.plot_second_order_analysis(energy=energy, nuclide_idx=i)
                    if fig:
                        html += '<div class="figure-container">'
                        html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Second Order">'
                        plt.close(fig)
                        html += '</div>'
        
        html += '</div>'  # report-container
        html += _generate_html_footer()
        
        return html
    
    def save_html(self, filepath: str, **kwargs) -> None:
        """
        Save HTML report to file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        **kwargs
            Arguments passed to generate_html()
        """
        html = self.generate_html(**kwargs)
        with open(filepath, 'w') as f:
            f.write(html)
        print(f"Report saved to: {filepath}")


# ============================================================================
# Serpent Sensitivity Report
# ============================================================================

@dataclass
class SerpentSensReport:
    """
    Generate comprehensive sensitivity analysis reports from Serpent results.
    
    This report class takes SensitivityFile objects from Serpent output parsing
    and generates:
    1. Sensitivity coefficient tables and plots
    2. Legendre moment analysis
    3. SDFData export for downstream UQ
    
    Parameters
    ----------
    sensitivity_file : SensitivityFile
        Parsed Serpent sensitivity output
    response : str or List[str]
        Response name(s) to analyze (e.g., 'sens_ratio_BIN_0')
    title : str, optional
        Report title
    """
    
    sensitivity_file: Any  # SensitivityFile from serpent module
    response: Union[str, List[str]]
    title: str = "Serpent Sensitivity Analysis Report"
    
    # Computed
    responses: List[str] = field(init=False)
    
    def __post_init__(self):
        """Initialize and validate."""
        if isinstance(self.response, str):
            self.responses = [self.response]
        else:
            self.responses = list(self.response)
    
    def summary(self) -> pd.DataFrame:
        """
        Generate summary DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Summary information
        """
        sf = self.sensitivity_file
        
        rows = [{
            'Materials': sf.n_materials,
            'Nuclides': sf.n_nuclides,
            'Perturbations': sf.n_perturbations,
            'Energy Bins': sf.n_energy_bins,
            'Responses': len(self.responses),
            'Reactions (MT)': ', '.join(map(str, sf.reactions[:5])) + ('...' if len(sf.reactions) > 5 else '')
        }]
        
        return pd.DataFrame(rows)
    
    def sensitivity_table(
        self,
        response: Optional[str] = None,
        material: Optional[Union[int, str]] = None,
        nuclide: Optional[Union[int, str]] = None,
        mt: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Generate integrated sensitivity table.
        
        Parameters
        ----------
        response : str, optional
            Response name. Uses first if None.
        material : int or str, optional
            Material filter
        nuclide : int or str, optional
            Nuclide filter
        mt : List[int], optional
            MT reaction filter
            
        Returns
        -------
        pd.DataFrame
            Integrated sensitivity values
        """
        sf = self.sensitivity_file
        resp = response or self.responses[0]
        
        rows = []
        
        # Get integrated sensitivities
        try:
            values, errors = sf.get_integrated(
                response=resp,
                mat=material,
                zai=nuclide,
                mt=mt
            )
        except Exception as e:
            warnings.warn(f"Could not get integrated sensitivities: {e}")
            return pd.DataFrame()
        
        # Build table based on returned shape
        # Shape should be (M, Z, P) or similar
        for mi, mat in enumerate(sf.materials):
            for zi, nuc in enumerate(sf.nuclides):
                for pi, pert in enumerate(sf.perturbations):
                    if mt is not None and pert.mt not in mt:
                        continue
                    
                    try:
                        val = values[mi, zi, pi]
                        err = errors[mi, zi, pi]
                    except (IndexError, KeyError):
                        continue
                    
                    if val == 0 and err == 0:
                        continue
                    
                    rows.append({
                        'Material': mat.name,
                        'Nuclide': _format_nuclide(nuc.zai),
                        'ZAI': nuc.zai,
                        'Reaction': pert.short_label or _format_reaction(pert.mt) if pert.mt else pert.raw_label,
                        'MT': pert.mt,
                        'Sensitivity': val,
                        'Rel. Error': err
                    })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Sensitivity', key=abs, ascending=False)
        return df
    
    def plot_sensitivities(
        self,
        response: Optional[str] = None,
        material: Optional[Union[int, str]] = None,
        nuclide: Optional[Union[int, str]] = None,
        mt: Optional[List[int]] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """
        Plot integrated sensitivity bar chart.
        
        Parameters
        ----------
        response : str, optional
            Response name
        material, nuclide, mt : filters
        top_n : int
            Number of top sensitivities
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
        """
        table = self.sensitivity_table(
            response=response, material=material, 
            nuclide=nuclide, mt=mt
        )
        
        if table.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No sensitivity data available',
                   ha='center', va='center', fontsize=14)
            return fig
        
        if len(table) > top_n:
            table = table.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [f"{row['Nuclide']} {row['Reaction']}" for _, row in table.iterrows()]
        values = table['Sensitivity'].values
        
        colors = ['#3498db' if v >= 0 else '#e74c3c' for v in values]
        
        ax.barh(range(len(values)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Integrated Sensitivity')
        ax.set_title(f'Top {len(table)} Sensitivity Coefficients')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_dependent(
        self,
        response: Optional[str] = None,
        material: Optional[Union[int, str]] = None,
        nuclide: Optional[Union[int, str]] = None,
        mt: Optional[List[int]] = None,
        per_lethargy: bool = True,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """
        Plot energy-dependent sensitivities.
        
        Uses the SensitivityFile's built-in plot method.
        """
        sf = self.sensitivity_file
        resp = response or self.responses[0]
        
        fig = sf.plot(
            response=resp,
            mat=material,
            zai=nuclide,
            mt=mt,
            per_lethargy=per_lethargy
        )
        
        return fig
    
    def to_sdf(
        self,
        response: Optional[str] = None,
        title: Optional[str] = None,
        material_filter: Optional[Union[str, List[str]]] = None,
        nuclide_filter: Optional[Union[int, List[int]]] = None,
        mt_filter: Optional[List[int]] = None
    ) -> SDFData:
        """
        Convert Serpent sensitivity data to SDF format.
        
        Parameters
        ----------
        response : str, optional
            Response name. Uses first if None.
        title : str, optional
            SDF title
        material_filter, nuclide_filter, mt_filter : filters
            
        Returns
        -------
        SDFData
        """
        # Import here to avoid circular imports
        from kika.sensitivities.sensitivity_processing import create_sdf_from_serpent
        
        return create_sdf_from_serpent(
            serpent_file=self.sensitivity_file,
            response_name=response or self.responses[0],
            title=title or self.title.replace(' ', '_'),
            material_filter=material_filter,
            nuclide_filter=nuclide_filter,
            mt_filter=mt_filter
        )
    
    def generate_html(
        self,
        response: Optional[str] = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate complete HTML report.
        
        Parameters
        ----------
        response : str, optional
            Response to analyze
        include_plots : bool
            Whether to include plots
            
        Returns
        -------
        str
            HTML document
        """
        html = _generate_html_header(self.title)
        
        html += '<div class="report-container">'
        html += f'<h1>{self.title}</h1>'
        
        # Summary
        html += '<h2>Summary</h2>'
        summary_df = self.summary()
        html += summary_df.to_html(index=False)
        
        # Metrics box
        sf = self.sensitivity_file
        html += '<div class="summary-box">'
        html += '<h3>Key Metrics</h3>'
        html += f'<div class="metric"><span class="metric-value">{sf.n_nuclides}</span>'
        html += '<span class="metric-label"> Nuclides</span></div>'
        html += f'<div class="metric"><span class="metric-value">{sf.n_energy_bins}</span>'
        html += '<span class="metric-label"> Energy Groups</span></div>'
        html += f'<div class="metric"><span class="metric-value">{len(sf.reactions)}</span>'
        html += '<span class="metric-label"> Reactions</span></div>'
        html += '</div>'
        
        # Sensitivity table
        html += '<h2>Integrated Sensitivities</h2>'
        sens_table = self.sensitivity_table(response=response)
        if not sens_table.empty:
            html += sens_table.head(20).to_html(index=False, float_format='%.4e')
            if len(sens_table) > 20:
                html += f'<p><em>Showing top 20 of {len(sens_table)} reactions</em></p>'
        else:
            html += '<p class="warning">No sensitivity data available.</p>'
        
        # Plots
        if include_plots and not sens_table.empty:
            html += '<h2>Sensitivity Profile</h2>'
            html += '<div class="figure-container">'
            fig = self.plot_sensitivities(response=response)
            html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Sensitivity Plot">'
            plt.close(fig)
            html += '</div>'
        
        html += '</div>'
        html += _generate_html_footer()
        
        return html
    
    def save_html(self, filepath: str, **kwargs) -> None:
        """Save HTML report to file."""
        html = self.generate_html(**kwargs)
        with open(filepath, 'w') as f:
            f.write(html)
        print(f"Report saved to: {filepath}")


# ============================================================================
# Comparison Report
# ============================================================================

@dataclass
class ComparisonReport:
    """
    Compare sensitivity data from multiple sources using SDF format.
    
    This is source-agnostic: it works with SDFData objects from any source
    (MCNP, Serpent, or parsed SDF files).
    
    Parameters
    ----------
    sdf_list : List[SDFData]
        List of SDF data objects to compare
    labels : List[str], optional
        Labels for each SDF dataset
    title : str, optional
        Report title
    """
    
    sdf_list: List[SDFData]
    labels: Optional[List[str]] = None
    title: str = "Sensitivity Comparison Report"
    
    # Computed
    _labels: List[str] = field(init=False)
    
    def __post_init__(self):
        """Initialize and validate."""
        if len(self.sdf_list) < 2:
            raise ValueError("ComparisonReport requires at least 2 SDF datasets")
        
        if self.labels is None:
            self._labels = [sdf.title for sdf in self.sdf_list]
        else:
            if len(self.labels) != len(self.sdf_list):
                raise ValueError("Number of labels must match number of SDF datasets")
            self._labels = list(self.labels)
    
    def find_common_reactions(self) -> List[Tuple[int, int]]:
        """
        Find (ZAID, MT) pairs common to all datasets.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (ZAID, MT) tuples present in all datasets
        """
        reaction_sets = []
        for sdf in self.sdf_list:
            reactions = {(r.zaid, r.mt) for r in sdf.data}
            reaction_sets.append(reactions)
        
        common = reaction_sets[0]
        for rs in reaction_sets[1:]:
            common = common.intersection(rs)
        
        return sorted(list(common))
    
    def comparison_table(
        self,
        reactions: Optional[List[Tuple[int, int]]] = None
    ) -> pd.DataFrame:
        """
        Generate comparison table for specified reactions.
        
        Parameters
        ----------
        reactions : List[Tuple[int, int]], optional
            List of (ZAID, MT) tuples. If None, uses common reactions.
            
        Returns
        -------
        pd.DataFrame
            Comparison table with sensitivities from each source
        """
        if reactions is None:
            reactions = self.find_common_reactions()
        
        rows = []
        for zaid, mt in reactions:
            row = {
                'Nuclide': _format_nuclide(zaid),
                'ZAID': zaid,
                'Reaction': _format_reaction(mt),
                'MT': mt
            }
            
            for i, (sdf, label) in enumerate(zip(self.sdf_list, self._labels)):
                # Find matching reaction
                matching = [r for r in sdf.data if r.zaid == zaid and r.mt == mt]
                if matching:
                    r = matching[0]
                    int_sens = sum(r.sensitivity)
                    row[f'{label}'] = int_sens
                else:
                    row[f'{label}'] = np.nan
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def difference_table(
        self,
        reference_idx: int = 0,
        reactions: Optional[List[Tuple[int, int]]] = None
    ) -> pd.DataFrame:
        """
        Calculate differences relative to a reference dataset.
        
        Parameters
        ----------
        reference_idx : int
            Index of reference dataset (default: 0)
        reactions : List[Tuple[int, int]], optional
            Reactions to compare
            
        Returns
        -------
        pd.DataFrame
            Table with differences (absolute and relative)
        """
        comp = self.comparison_table(reactions)
        ref_label = self._labels[reference_idx]
        
        for i, label in enumerate(self._labels):
            if i == reference_idx:
                continue
            
            diff_col = f'{label} - {ref_label}'
            rel_diff_col = f'{label} vs {ref_label} (%)'
            
            comp[diff_col] = comp[label] - comp[ref_label]
            comp[rel_diff_col] = np.where(
                comp[ref_label] != 0,
                (comp[label] - comp[ref_label]) / np.abs(comp[ref_label]) * 100,
                np.nan
            )
        
        return comp
    
    def plot_comparison(
        self,
        reactions: Optional[List[Tuple[int, int]]] = None,
        top_n: int = 10,
        figsize: Tuple[int, int] = (14, 8)
    ) -> Figure:
        """
        Plot side-by-side comparison of sensitivities.
        
        Parameters
        ----------
        reactions : List[Tuple[int, int]], optional
            Reactions to compare
        top_n : int
            Number of reactions to plot
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
        """
        comp = self.comparison_table(reactions)
        
        if comp.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No common reactions found',
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Sort by maximum absolute sensitivity across all sources
        sens_cols = [col for col in comp.columns if col in self._labels]
        comp['max_abs'] = comp[sens_cols].abs().max(axis=1)
        comp = comp.sort_values('max_abs', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(comp))
        width = 0.8 / len(self._labels)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self._labels)))
        
        for i, label in enumerate(self._labels):
            offset = (i - len(self._labels)/2 + 0.5) * width
            ax.bar(x + offset, comp[label].values, width, 
                  label=label, color=colors[i], alpha=0.8)
        
        ax.set_xticks(x)
        labels = [f"{row['Nuclide']}\n{row['Reaction']}" for _, row in comp.iterrows()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Integrated Sensitivity')
        ax.set_title(f'Sensitivity Comparison (Top {len(comp)} Reactions)')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        reactions: Optional[List[Tuple[int, int]]] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> Figure:
        """
        Scatter plot comparing two datasets.
        
        Parameters
        ----------
        x_idx, y_idx : int
            Indices of datasets to compare
        reactions : List[Tuple[int, int]], optional
            Reactions to include
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
        """
        comp = self.comparison_table(reactions)
        
        x_label = self._labels[x_idx]
        y_label = self._labels[y_idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_vals = comp[x_label].values
        y_vals = comp[y_label].values
        
        # Remove NaN pairs
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        
        ax.scatter(x_vals, y_vals, alpha=0.7, s=50)
        
        # Add identity line
        lims = [
            min(min(x_vals), min(y_vals)),
            max(max(x_vals), max(y_vals))
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
        
        # Calculate R² if possible
        if len(x_vals) > 1:
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            r2 = correlation ** 2
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Sensitivity Comparison: {x_label} vs {y_label}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_html(self, include_plots: bool = True) -> str:
        """Generate complete HTML comparison report."""
        html = _generate_html_header(self.title)
        
        html += '<div class="report-container">'
        html += f'<h1>{self.title}</h1>'
        
        # Summary
        html += '<h2>Datasets</h2>'
        html += '<table><tr><th>#</th><th>Label</th><th>Title</th><th>Reactions</th></tr>'
        for i, (sdf, label) in enumerate(zip(self.sdf_list, self._labels)):
            html += f'<tr><td>{i+1}</td><td>{label}</td><td>{sdf.title}</td><td>{len(sdf.data)}</td></tr>'
        html += '</table>'
        
        # Common reactions
        common = self.find_common_reactions()
        html += '<div class="info">'
        html += f'<strong>{len(common)} common reactions</strong> found across all datasets.'
        html += '</div>'
        
        # Comparison table
        html += '<h2>Sensitivity Comparison</h2>'
        comp_table = self.comparison_table()
        if not comp_table.empty:
            html += comp_table.to_html(index=False, float_format='%.4e')
        
        # Difference table
        html += '<h2>Differences (vs first dataset)</h2>'
        diff_table = self.difference_table()
        if not diff_table.empty:
            # Only show relevant columns
            cols = ['Nuclide', 'Reaction'] + [c for c in diff_table.columns if '%' in c]
            html += diff_table[cols].to_html(index=False, float_format='%.2f')
        
        # Plots
        if include_plots and not comp_table.empty:
            html += '<h2>Comparison Plots</h2>'
            
            html += '<h3>Bar Chart Comparison</h3>'
            html += '<div class="figure-container">'
            fig = self.plot_comparison()
            html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Comparison">'
            plt.close(fig)
            html += '</div>'
            
            if len(self.sdf_list) == 2:
                html += '<h3>Scatter Plot</h3>'
                html += '<div class="figure-container">'
                fig = self.plot_scatter()
                html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Scatter">'
                plt.close(fig)
                html += '</div>'
        
        html += '</div>'
        html += _generate_html_footer()
        
        return html
    
    def save_html(self, filepath: str, **kwargs) -> None:
        """Save HTML report to file."""
        html = self.generate_html(**kwargs)
        with open(filepath, 'w') as f:
            f.write(html)
        print(f"Report saved to: {filepath}")


# ============================================================================
# UQ Report
# ============================================================================

@dataclass
class UQReport:
    """
    Uncertainty Quantification report using the sandwich formula.
    
    This wraps sandwich_uncertainty_propagation and generates comprehensive
    reports with contribution breakdowns and visualizations.
    
    Parameters
    ----------
    sdf_data : SDFData
        Sensitivity data in SDF format
    cov_mat : CovMat or List[CovMat], optional
        Cross-section covariance matrices
    legendre_cov_mat : MGMF34CovMat or List[MGMF34CovMat], optional
        Legendre moment covariance matrices
    title : str, optional
        Report title
    verbose : bool
        Print detailed information during calculation
    """
    
    sdf_data: SDFData
    cov_mat: Optional[Any] = None  # CovMat or List[CovMat]
    legendre_cov_mat: Optional[Any] = None  # MGMF34CovMat or List[MGMF34CovMat]
    title: str = "Uncertainty Quantification Report"
    verbose: bool = False
    
    # Computed
    result: Any = field(init=False, default=None)  # UncertaintyResult
    _computed: bool = field(init=False, default=False)
    
    def compute(self) -> 'UQReport':
        """
        Run the sandwich uncertainty propagation.
        
        Returns
        -------
        UQReport
            Self (for method chaining)
        """
        from kika.UQ.sandwich import sandwich_uncertainty_propagation
        
        self.result = sandwich_uncertainty_propagation(
            sdf_data=self.sdf_data,
            cov_mat=self.cov_mat,
            legendre_cov_mat=self.legendre_cov_mat,
            verbose=self.verbose
        )
        self._computed = True
        return self
    
    def _ensure_computed(self):
        """Ensure calculation has been run."""
        if not self._computed:
            self.compute()
    
    def summary(self) -> pd.DataFrame:
        """
        Generate summary DataFrame.
        
        Returns
        -------
        pd.DataFrame
        """
        self._ensure_computed()
        
        r = self.result
        
        # Calculate combined uncertainty
        stat_abs = r.response_error * abs(r.response_value)
        nuc_abs = r.relative_uncertainty * abs(r.response_value)
        total_abs = np.sqrt(stat_abs**2 + nuc_abs**2)
        total_rel = total_abs / abs(r.response_value) * 100 if r.response_value != 0 else 0
        
        rows = [{
            'Metric': 'Response Value',
            'Value': f'{r.response_value:.6e}'
        }, {
            'Metric': 'Statistical Uncertainty (Monte Carlo)',
            'Value': f'± {r.response_error*100:.3f}%'
        }, {
            'Metric': 'Nuclear Data Uncertainty (Sandwich)',
            'Value': f'± {r.relative_uncertainty*100:.3f}%'
        }, {
            'Metric': 'Total Combined Uncertainty',
            'Value': f'± {total_rel:.3f}%'
        }, {
            'Metric': 'Reactions Included',
            'Value': str(r.n_reactions)
        }, {
            'Metric': 'Energy Groups',
            'Value': str(r.n_energy_groups)
        }]
        
        return pd.DataFrame(rows)
    
    def contribution_table(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Generate contribution breakdown table.
        
        Parameters
        ----------
        top_n : int, optional
            Number of top contributors to show
            
        Returns
        -------
        pd.DataFrame
        """
        self._ensure_computed()
        
        contributions = self.result.contributions
        if top_n:
            contributions = contributions[:top_n]
        
        rows = []
        for c in contributions:
            rows.append({
                'Nuclide': c.nuclide,
                'Reaction': c.reaction_name,
                'MT': c.mt,
                'Variance Contribution': c.variance_contribution,
                'Uncertainty (σ)': c.uncertainty_contribution,
                'Relative (%)': c.relative_contribution * 100
            })
        
        return pd.DataFrame(rows)
    
    def plot_contributions(
        self,
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 6)
    ) -> Figure:
        """
        Plot contribution bar chart.
        
        Parameters
        ----------
        top_n : int
            Number of top contributors
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
        """
        self._ensure_computed()
        
        table = self.contribution_table(top_n=top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        labels = [f"{row['Nuclide']} {row['Reaction']}" for _, row in table.iterrows()]
        values = table['Relative (%)'].values
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(values)))
        
        bars = ax.barh(range(len(values)), values, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=10)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Contribution to Total Variance (%)')
        ax.set_title(f'Top {len(table)} Uncertainty Contributors')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(values) * 1.15)
        
        plt.tight_layout()
        return fig
    
    def plot_pie(
        self,
        top_n: int = 8,
        figsize: Tuple[int, int] = (10, 10)
    ) -> Figure:
        """
        Plot pie chart of contributions.
        
        Parameters
        ----------
        top_n : int
            Number of categories (rest grouped as "Other")
        figsize : tuple
            Figure size
            
        Returns
        -------
        Figure
        """
        self._ensure_computed()
        
        table = self.contribution_table()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(table) > top_n:
            top = table.head(top_n - 1)
            other_pct = table.iloc[top_n-1:]['Relative (%)'].sum()
            
            labels = [f"{row['Nuclide']} {row['Reaction']}" for _, row in top.iterrows()]
            values = list(top['Relative (%)'].values) + [other_pct]
            labels.append('Other')
        else:
            labels = [f"{row['Nuclide']} {row['Reaction']}" for _, row in table.iterrows()]
            values = table['Relative (%)'].values
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(values)))
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors, pctdistance=0.75
        )
        
        ax.set_title('Uncertainty Contribution Breakdown')
        
        plt.tight_layout()
        return fig
    
    def generate_html(self, include_plots: bool = True) -> str:
        """Generate complete HTML UQ report."""
        self._ensure_computed()
        
        html = _generate_html_header(self.title)
        
        html += '<div class="report-container">'
        html += f'<h1>{self.title}</h1>'
        
        # Summary box
        r = self.result
        stat_pct = r.response_error * 100
        nuc_pct = r.relative_uncertainty * 100
        total_pct = np.sqrt(stat_pct**2 + nuc_pct**2)
        
        html += '<div class="summary-box">'
        html += '<h3>Uncertainty Summary</h3>'
        html += f'<div class="metric"><span class="metric-value">{r.response_value:.4e}</span>'
        html += '<span class="metric-label"> Response Value</span></div>'
        html += f'<div class="metric"><span class="metric-value">± {total_pct:.2f}%</span>'
        html += '<span class="metric-label"> Total Uncertainty</span></div>'
        html += f'<div class="metric"><span class="metric-value">± {nuc_pct:.2f}%</span>'
        html += '<span class="metric-label"> Nuclear Data</span></div>'
        html += f'<div class="metric"><span class="metric-value">± {stat_pct:.2f}%</span>'
        html += '<span class="metric-label"> Statistical</span></div>'
        html += '</div>'
        
        # Summary table
        html += '<h2>Results Summary</h2>'
        html += self.summary().to_html(index=False)
        
        # Contribution table
        html += '<h2>Uncertainty Contributors</h2>'
        contrib_table = self.contribution_table(top_n=20)
        html += contrib_table.to_html(index=False, float_format='%.4e')
        
        # Plots
        if include_plots:
            html += '<h2>Visualizations</h2>'
            
            html += '<h3>Contribution Bar Chart</h3>'
            html += '<div class="figure-container">'
            fig = self.plot_contributions()
            html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Contributions">'
            plt.close(fig)
            html += '</div>'
            
            html += '<h3>Pie Chart</h3>'
            html += '<div class="figure-container">'
            fig = self.plot_pie()
            html += f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" alt="Pie Chart">'
            plt.close(fig)
            html += '</div>'
        
        # Full UncertaintyResult repr
        html += '<h2>Detailed Calculation Summary</h2>'
        html += f'<pre>{str(r)}</pre>'
        
        html += '</div>'
        html += _generate_html_footer()
        
        return html
    
    def save_html(self, filepath: str, **kwargs) -> None:
        """Save HTML report to file."""
        html = self.generate_html(**kwargs)
        with open(filepath, 'w') as f:
            f.write(html)
        print(f"Report saved to: {filepath}")


# ============================================================================
# Convenience function for quick analysis
# ============================================================================

def quick_sensitivity_report(
    source: Union[SensitivityData, List[SensitivityData], 'SensitivityFile', SDFData],
    source_type: str = 'auto',
    response: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None
) -> Union[MCNPPertReport, SerpentSensReport, ComparisonReport]:
    """
    Convenience function to quickly generate a sensitivity report.
    
    Parameters
    ----------
    source : SensitivityData, List[SensitivityData], SensitivityFile, or SDFData
        Input sensitivity data
    source_type : str
        'mcnp', 'serpent', 'sdf', or 'auto' (auto-detect)
    response : str, optional
        Response name (for Serpent)
    title : str, optional
        Report title
    output_path : str, optional
        If provided, saves HTML report to this path
        
    Returns
    -------
    Report object
        MCNPPertReport, SerpentSensReport, or comparison based on input
    """
    # Auto-detect source type
    if source_type == 'auto':
        if isinstance(source, SensitivityData):
            source_type = 'mcnp'
        elif isinstance(source, list) and source and isinstance(source[0], SensitivityData):
            source_type = 'mcnp'
        elif isinstance(source, SDFData):
            source_type = 'sdf'
        elif hasattr(source, 'n_perturbations'):  # SensitivityFile
            source_type = 'serpent'
        else:
            raise ValueError(f"Cannot auto-detect source type for {type(source)}")
    
    # Create appropriate report
    if source_type == 'mcnp':
        report = MCNPPertReport(
            sensitivity_data=source,
            title=title or "MCNP PERT Sensitivity Analysis"
        )
    elif source_type == 'serpent':
        if response is None:
            raise ValueError("response parameter required for Serpent data")
        report = SerpentSensReport(
            sensitivity_file=source,
            response=response,
            title=title or "Serpent Sensitivity Analysis"
        )
    elif source_type == 'sdf':
        warnings.warn("SDFData alone cannot generate a full report. Consider using MCNPPertReport or SerpentSensReport.")
        return None
    else:
        raise ValueError(f"Unknown source_type: {source_type}")
    
    # Save if path provided
    if output_path:
        report.save_html(output_path)
    
    return report
