"""
Utility functions for rendering heatmap plots.

This module contains helper functions for energy tick formatting, uncertainty panels,
and other heatmap-specific visualizations used by PlotBuilder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Dict, Optional


def setup_energy_group_ticks(
    ax: plt.Axes,
    mts_sorted: List[int],
    G: int,
    matrix_size: int
) -> None:
    """
    Helper function to setup energy group ticks and labels for covariance heatmaps.
    
    - Tick lines are drawn only for 10th bin boundaries.
    - Labels for 10th/20th bins (not G) are at the tick line.
    - Label for G is always shown, centered in G's cell.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes for the heatmap
    mts_sorted : list of int
        Sorted list of MT numbers
    G : int
        Number of energy groups per MT
    matrix_size : int
        Total size of the matrix
    """
    tick_line_positions = []  # For drawing physical tick lines
    label_info_list = []      # For storing {'pos': coordinate, 'label': string, 'mt_idx': int}
    
    num_mts = len(mts_sorted)

    # Determine label frequency based on number of MTs
    label_frequency = 20 if num_mts > 4 else 10

    for i in range(len(mts_sorted)):
        block_start = i * G
        
        # Add tick lines for every 10th energy group
        for g in range(10, G, 10):
            pos = block_start + g - 0.5  # Tick at bin boundary
            tick_line_positions.append(pos)
            
            # Add label at label_frequency intervals (10 or 20)
            if g % label_frequency == 0:
                label_info_list.append({
                    'pos': block_start + g,  # Center of bin
                    'label': str(g),
                    'mt_idx': i
                })
        
        # Always add label for G (maximum group), centered in its cell
        if G not in [g for g in range(10, G, 10)]:  # If not already added
            label_info_list.append({
                'pos': block_start + G - 0.5,  # Center of last bin
                'label': str(G),
                'mt_idx': i
            })

    # Ensure uniqueness and sort positions
    tick_line_positions = sorted(list(set(tick_line_positions)))
    
    # Determine all unique positions involved for setting Matplotlib ticks
    all_involved_positions = sorted(list(set(tick_line_positions + [item['pos'] for item in label_info_list])))
    
    if all_involved_positions:
        ax.set_xticks(all_involved_positions)
        ax.set_yticks(all_involved_positions)
        ax.set_xticklabels([])  # Will add custom labels below
        ax.set_yticklabels([])
        
    # Custom draw the tick lines
    if tick_line_positions:
        for pos in tick_line_positions:
            # Horizontal line
            ax.axhline(pos, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
            # Vertical line
            ax.axvline(pos, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
        
    # Label positioning configuration
    label_offset_config = {
        1: 1.5, 2: 3.5, 3: 4.8, 4: 6.5, 5: 8.5, 6: 10.5, 7: 12.5,
    }
    bottom_offset = label_offset_config.get(num_mts, label_offset_config[7])
    
    # Add labels for the selected ticks/positions
    if label_info_list:
        for item in label_info_list:
            pos = item['pos']
            label = item['label']
            mt_idx = item['mt_idx']
            
            # Bottom labels (x-axis)
            ax.text(
                pos, -bottom_offset, label,
                ha='center', va='top',
                fontsize=9, color='gray'
            )
            
            # Left labels (y-axis)
            ax.text(
                -bottom_offset, pos, label,
                ha='right', va='center',
                fontsize=9, color='gray'
            )


def setup_energy_group_ticks_single_block(ax: plt.Axes, G: int) -> None:
    """
    Helper function to setup energy group ticks and labels for single off-diagonal block.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes for the heatmap
    G : int
        Number of energy groups
    """
    tick_line_positions = []
    label_info_list = []
    
    # Determine label frequency
    label_frequency = 20 if G > 100 else 10

    # Add tick lines and labels for this single block
    for g in range(10, G, 10):
        pos = g - 0.5  # Tick at bin boundary
        tick_line_positions.append(pos)
        
        if g % label_frequency == 0:
            label_info_list.append({
                'pos': g,  # Center of bin
                'label': str(g)
            })
    
    # Always add label for G
    if G not in [g for g in range(10, G, 10)]:
        label_info_list.append({
            'pos': G - 0.5,
            'label': str(G)
        })

    # Set ticks
    all_involved_positions = sorted(list(set(tick_line_positions + [item['pos'] for item in label_info_list])))
    
    if all_involved_positions:
        ax.set_xticks(all_involved_positions)
        ax.set_yticks(all_involved_positions)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    # Draw tick lines
    if tick_line_positions:
        for pos in tick_line_positions:
            ax.axhline(pos, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
            ax.axvline(pos, color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Add labels
    bottom_offset = 3.5
    if label_info_list:
        for item in label_info_list:
            pos = item['pos']
            label = item['label']
            
            ax.text(
                pos, -bottom_offset, label,
                ha='center', va='top',
                fontsize=9, color='gray'
            )
            ax.text(
                -bottom_offset, pos, label,
                ha='right', va='center',
                fontsize=9, color='gray'
            )


def format_uncertainty_ticks(ax: plt.Axes, sigma_pct: np.ndarray) -> None:
    """
    Format y-axis ticks for uncertainty panels.
    
    Parameters
    ----------
    ax : plt.Axes
        Uncertainty panel axes
    sigma_pct : np.ndarray
        Uncertainty percentage values
    """
    if len(sigma_pct) == 0 or np.all(np.isnan(sigma_pct)):
        return
    
    # Calculate nice tick values
    max_val = np.nanmax(sigma_pct)
    if max_val < 1:
        tick_interval = 0.2
    elif max_val < 5:
        tick_interval = 1
    elif max_val < 20:
        tick_interval = 5
    elif max_val < 50:
        tick_interval = 10
    else:
        tick_interval = 20
    
    # Generate ticks from 0 to max_val
    ticks = []
    val = 0
    while val <= max_val * 1.1:
        ticks.append(val)
        val += tick_interval
    
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(t)}' for t in ticks], fontsize=9)


def add_mt_labels_to_heatmap(
    ax: plt.Axes,
    mts_sorted: List[int],
    G: int,
    is_diagonal: bool,
    row_mt: Optional[int] = None,
    col_mt: Optional[int] = None
) -> None:
    """
    Add MT number labels to heatmap axes.
    
    Parameters
    ----------
    ax : plt.Axes
        Heatmap axes
    mts_sorted : list of int
        Sorted list of MT numbers (for diagonal blocks)
    G : int
        Number of energy groups per MT
    is_diagonal : bool
        Whether this is a diagonal block
    row_mt : int, optional
        Row MT for off-diagonal block
    col_mt : int, optional
        Column MT for off-diagonal block
    """
    if is_diagonal:
        # Diagonal blocks: multiple MTs
        mt_tick_positions = [(i + 0.5) * G for i in range(len(mts_sorted))]
        mt_labels = [str(mt) for mt in mts_sorted]
        
        ax.set_xticks(mt_tick_positions, minor=False)
        ax.set_xticklabels(mt_labels)
        ax.set_yticks(mt_tick_positions, minor=False)
        ax.set_yticklabels(mt_labels)
    else:
        # Off-diagonal block: single MT per axis
        center_pos = G / 2
        
        if col_mt is not None:
            ax.set_xticks([center_pos], minor=False)
            ax.set_xticklabels([str(col_mt)])
        
        if row_mt is not None:
            ax.set_yticks([center_pos], minor=False)
            ax.set_yticklabels([str(row_mt)])
