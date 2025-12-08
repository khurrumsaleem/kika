"""
Backend and notebook detection utilities for matplotlib.

This module provides utilities for detecting and configuring matplotlib
backends in different environments (notebooks, interactive, etc.).
"""

import matplotlib.pyplot as plt


def _is_notebook() -> bool:
    """
    Check if code is running in a Jupyter notebook.
    
    Returns
    -------
    bool
        True if running in a notebook, False otherwise
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False
        # Check if we're in a notebook environment
        return hasattr(ipython, 'kernel')
    except ImportError:
        return False


def _detect_interactive_backend() -> bool:
    """
    Detect if matplotlib is using an interactive backend suitable for widgets.
    
    Returns
    -------
    bool
        True if using widget/interactive backend, False otherwise
    """
    current_backend = plt.get_backend()
    
    # Widget backends that support full interactivity
    widget_backends = [
        'module://ipympl.backend_nbagg',  # ipympl widget backend
        'widget',                         # Alternative ipympl backend name
        'Qt5Agg', 'Qt4Agg', 'TkAgg',     # Desktop interactive backends
        'nbAgg'                           # Notebook backend (limited interactivity)
    ]
    
    # Check if current backend supports interactivity
    is_interactive = current_backend in widget_backends
    
    # Additional check for ipympl specifically
    if current_backend in ['module://ipympl.backend_nbagg', 'widget']:
        try:
            import ipympl
            return True
        except ImportError:
            return False
    
    return is_interactive


def _setup_notebook_backend() -> tuple[bool, str]:
    """
    Setup appropriate matplotlib backend for notebooks.
    
    Returns
    -------
    tuple of (bool, str)
        (success, backend_description)
        success: True if interactive backend is available
        backend_description: Description of the backend being used
    """
    if not _is_notebook():
        return False, "Not in notebook"
    
    current_backend = plt.get_backend()
    
    # If already using widget backend, don't change it
    if current_backend in ['module://ipympl.backend_nbagg', 'widget']:
        try:
            import ipympl
            return True, f"ipympl widget backend ({current_backend})"
        except ImportError:
            pass
    
    try:
        # Try ipympl first (recommended for interactive plots)
        import ipympl
        if current_backend != 'module://ipympl.backend_nbagg':
            plt.switch_backend('widget')
        return True, "ipympl widget backend"
    except ImportError:
        try:
            # Fallback to notebook backend
            if current_backend != 'nbAgg':
                plt.switch_backend('notebook')
            return True, "notebook backend"
        except Exception:
            # Last resort - inline
            plt.switch_backend('inline')
            return False, "inline backend (no interactivity)"


def _configure_figure_interactivity(fig, interactive_mode: bool) -> None:
    """
    Configure figure interactivity based on the mode.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object
    interactive_mode : bool
        Whether to enable interactive features
    """
    if interactive_mode:
        # Enable interactive features for widget backends
        try:
            # Enable toolbar if available
            if hasattr(fig.canvas, 'toolbar_visible'):
                fig.canvas.toolbar_visible = True
        except Exception:
            # Silently fall back if interactive features aren't available
            pass
    else:
        # Disable interactive features for static backends
        try:
            if hasattr(fig.canvas, 'toolbar_visible'):
                fig.canvas.toolbar_visible = False
        except Exception:
            pass
