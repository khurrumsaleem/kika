"""
MF34 (Angular Distribution Covariance) creation and writing utilities.

This module provides functions to:
1. Create MF34MT objects from Legendre coefficient covariance matrices
2. Write MF34 sections to ENDF files
3. Support for LB=5 format (full symmetric matrix storage)

The covariance matrix is expected to be organized with:
- Rows/columns representing (energy, Legendre order) pairs
- Layout: [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

Example:
    >>> from kika.endf.writers import create_mf34_from_covariance, write_mf34_to_file
    >>>
    >>> # Create MF34 from covariance matrix
    >>> mf34 = create_mf34_from_covariance(
    ...     cov_matrix=cov,
    ...     energy_grid_ev=energy_boundaries,
    ...     max_order=8,
    ...     za=26056.0,
    ...     awr=55.47,
    ...     mat=2631,
    ...     mt=2,
    ... )
    >>>
    >>> # Write to ENDF file
    >>> write_mf34_to_file('base.endf', mf34, 'output.endf')
"""
from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import numpy as np

from ..classes.mf34.mf34 import (
    MF34MT,
    Subsection,
    SubSubsection,
    SubSubsectionRecord,
)

if TYPE_CHECKING:
    from pathlib import Path


def create_mf34_from_covariance(
    cov_matrix: np.ndarray,
    energy_grid_ev: np.ndarray,
    max_order: int,
    za: float,
    awr: float,
    mat: int,
    mt: int,
    ltt: int = 1,
    mt1: Optional[int] = None,
    frame: str = "same-as-MF4",
) -> MF34MT:
    """
    Create MF34MT object from Legendre coefficient covariance matrix.

    This function constructs an MF34 (Angular Distribution Covariance) section
    from a pre-computed covariance matrix of Legendre polynomial coefficients.
    The output uses LB=5 format with symmetric upper-triangle storage.

    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix with shape (N_energies * L_max, N_energies * L_max).
        The matrix should be organized with Legendre orders as the fast index:
        index = i_energy * max_order + (l - 1)

        Layout: [a_1(E_1), a_2(E_1), ..., a_L(E_1), a_1(E_2), ..., a_L(E_N)]

    energy_grid_ev : np.ndarray
        Energy boundary points in eV. For N_energies energy intervals,
        provide N_energies + 1 boundary points.

    max_order : int
        Maximum Legendre order (L_max). The covariance includes orders 1 to L_max
        (following ENDF convention where a_0 is implicit from normalization).

    za : float
        ZA identifier (1000*Z + A). For example, 26056.0 for Fe-56.

    awr : float
        Atomic weight ratio (mass of target in neutron mass units).

    mat : int
        MAT number for the material in the ENDF file.

    mt : int
        MT reaction number. Common values:
        - MT=2: Elastic scattering
        - MT=18: Total fission

    ltt : int, default 1
        LTT flag indicating Legendre representation:
        - LTT=1: Coefficients start with a_1 (standard ENDF convention)
        - LTT=2: Coefficients start with a_0

    mt1 : int, optional
        MT1 for cross-correlation between different reactions.
        If None, defaults to mt (self-correlation of the same reaction).

    frame : str, default "same-as-MF4"
        Reference frame for angular distributions:
        - "same-as-MF4": Use same frame as MF4 section (LCT=0)
        - "LAB": Laboratory frame (LCT=1)
        - "CM": Center-of-mass frame (LCT=2)

    Returns
    -------
    MF34MT
        MF34MT object ready for serialization to ENDF format using str(mf34).

    Raises
    ------
    ValueError
        If covariance matrix dimensions don't match expected size.

    Notes
    -----
    The output uses LB=5 format (full matrix storage) with LS=1 (symmetric,
    upper-triangle only). For each (L, L1) Legendre pair, a sub-subsection
    is created containing the energy-energy covariance block.

    Only the upper triangle of (L, L1) pairs is stored since the covariance
    is symmetric: Cov(a_L, a_{L1}) = Cov(a_{L1}, a_L).

    Examples
    --------
    Create MF34 for Fe-56 elastic scattering with 8 Legendre orders:

    >>> import numpy as np
    >>> # Suppose we have 10 energy intervals and 8 Legendre orders
    >>> n_energies = 10
    >>> max_order = 8
    >>> cov = np.eye(n_energies * max_order) * 0.01  # Example diagonal covariance
    >>> energy_grid = np.linspace(1e6, 20e6, n_energies + 1)  # eV
    >>>
    >>> mf34 = create_mf34_from_covariance(
    ...     cov_matrix=cov,
    ...     energy_grid_ev=energy_grid,
    ...     max_order=8,
    ...     za=26056.0,
    ...     awr=55.47,
    ...     mat=2631,
    ...     mt=2,
    ... )
    >>>
    >>> # Get ENDF-formatted string
    >>> endf_text = str(mf34)
    """
    if mt1 is None:
        mt1 = mt

    n_energies = len(energy_grid_ev) - 1  # Number of energy intervals

    # Validate covariance matrix dimensions
    expected_size = n_energies * max_order
    if cov_matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} doesn't match "
            f"expected ({expected_size}, {expected_size}) for "
            f"{n_energies} energy intervals and {max_order} Legendre orders"
        )

    # Create MF34MT structure
    mf34 = MF34MT(number=mt)
    mf34._za = za
    mf34._awr = awr
    mf34._mat = mat
    mf34._ltt = ltt
    mf34._mf = 34

    # Create subsection for MT1 correlation
    subsection = Subsection()
    subsection.mt1 = mt1
    subsection.nl = max_order   # Number of Legendre coefficients for MT
    subsection.nl1 = max_order  # Number of Legendre coefficients for MT1
    subsection.mat1 = 0.0

    # LCT value based on frame
    lct_map = {"same-as-MF4": 0, "LAB": 1, "CM": 2}
    lct = lct_map.get(frame, 0)

    # Create sub-subsections for each (L, L1) pair
    # Only upper triangle: L <= L1 (symmetric covariance)
    for l in range(1, max_order + 1):
        for l1 in range(l, max_order + 1):
            sub_subsec = SubSubsection()
            sub_subsec.l = l
            sub_subsec.l1 = l1
            sub_subsec.lct = lct
            sub_subsec.ni = 1  # One LIST record

            # Extract sub-matrix for this (L, L1) block
            # Indices: for each energy E_i, coeff a_l is at index: i * max_order + (l-1)
            row_indices = [i * max_order + (l - 1) for i in range(n_energies)]
            col_indices = [i * max_order + (l1 - 1) for i in range(n_energies)]

            sub_matrix = cov_matrix[np.ix_(row_indices, col_indices)]

            # Create LB=5 record with LS=1 (symmetric upper triangle storage)
            record = SubSubsectionRecord()
            record.ls = 1  # Symmetric storage (upper triangle)
            record.lb = 5
            record.ne = len(energy_grid_ev)  # Number of energy boundary points

            # Store energies (boundary points)
            record.energies = list(energy_grid_ev)

            # Store matrix values - upper triangle, row-wise
            # For M x M matrix: F(0,0), F(0,1), ..., F(0,M-1), F(1,1), ..., F(M-1,M-1)
            matrix_values = []
            for k in range(n_energies):
                for l_idx in range(k, n_energies):
                    matrix_values.append(float(sub_matrix[k, l_idx]))

            record.matrix = matrix_values
            record.nt = len(energy_grid_ev) + len(matrix_values)

            sub_subsec.records = [record]
            subsection.sub_subsections.append(sub_subsec)

    mf34._nmt1 = 1  # One subsection
    mf34._subsections = [subsection]

    return mf34


def write_mf34_to_file(
    source_endf: str,
    mf34: MF34MT,
    output_path: str,
    replace_existing: bool = True,
) -> str:
    """
    Write MF34 section to an ENDF file.

    This function takes a source ENDF file as a template and either replaces
    an existing MF34 section or inserts a new one before the MEND marker.

    Parameters
    ----------
    source_endf : str or Path
        Path to source ENDF file that serves as the template.
        All content except MF34 will be preserved.

    mf34 : MF34MT
        MF34MT object to write. Use create_mf34_from_covariance() to create this.

    output_path : str or Path
        Path for the output ENDF file.

    replace_existing : bool, default True
        If True and MF34 already exists in source, replace it.
        If False and MF34 exists, raise FileExistsError.

    Returns
    -------
    str
        Path to the output file.

    Raises
    ------
    FileNotFoundError
        If source_endf file doesn't exist.
    FileExistsError
        If MF34 exists in source and replace_existing=False.

    Examples
    --------
    Add MF34 to an ENDF file:

    >>> mf34 = create_mf34_from_covariance(...)
    >>> write_mf34_to_file('evaluation.endf', mf34, 'evaluation_with_cov.endf')
    'evaluation_with_cov.endf'
    """
    # Read source file
    with open(source_endf, 'r') as f:
        lines = f.readlines()

    # Find MF34 boundaries if it exists
    mf34_start, mf34_end = _find_mf34_boundaries(lines)
    has_mf34 = mf34_start is not None

    if has_mf34 and not replace_existing:
        raise FileExistsError(
            f"MF34 already exists in {source_endf}. "
            f"Set replace_existing=True to replace it."
        )

    # Convert MF34MT to string
    mf34_content = str(mf34)
    mf34_lines = [line + '\n' for line in mf34_content.split('\n') if line.strip()]

    if has_mf34:
        # Replace existing MF34
        new_lines = lines[:mf34_start] + mf34_lines + lines[mf34_end:]
    else:
        # Insert MF34 before MEND marker
        insert_idx = _find_mend_marker(lines)
        new_lines = lines[:insert_idx] + mf34_lines + lines[insert_idx:]

    # Write output
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

    return output_path


def _find_mf34_boundaries(lines: List[str]) -> tuple:
    """
    Find start and end line indices of MF34 section.

    Parameters
    ----------
    lines : List[str]
        Lines from ENDF file.

    Returns
    -------
    tuple
        (start_index, end_index) or (None, None) if MF34 not found.
    """
    mf34_start = None
    mf34_end = None

    for i, line in enumerate(lines):
        if len(line) >= 75:
            try:
                mf = int(line[70:72].strip() or '0')
                if mf == 34:
                    if mf34_start is None:
                        mf34_start = i
                    mf34_end = i + 1
            except ValueError:
                continue

    return mf34_start, mf34_end


def _find_mend_marker(lines: List[str]) -> int:
    """
    Find insertion point (line index before MEND marker).

    The MEND marker is identified by a line with MAT > 0, MF = 0, MT = 0.

    Parameters
    ----------
    lines : List[str]
        Lines from ENDF file.

    Returns
    -------
    int
        Line index where MF34 should be inserted.
    """
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if len(line) >= 75:
            try:
                mat = int(line[66:70].strip() or '0')
                mf = int(line[70:72].strip() or '0')
                mt = int(line[72:75].strip() or '0')
                if mat > 0 and mf == 0 and mt == 0:
                    return i
            except ValueError:
                continue
    return len(lines)
