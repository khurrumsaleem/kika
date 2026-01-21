"""
EXFOR data loading functions.

This module provides the main I/O functions for loading EXFOR data from
JSON files or the X4Pro database, and organizing datasets by energy.

Supports multiple data sources:
- JSON files: Local JSON files in standardized format
- Database: X4Pro SQLite database (full 2025 version with JSON schema)
- Auto: Database with JSON fallback for missing entries
"""

import os
import glob
import math
from typing import Dict, List, Optional, Tuple, Union

from kika.exfor._constants import ENERGY_MATCH_ABS_TOL
from kika.exfor.config import get_db_path
from kika.exfor.angular_distribution import ExforAngularDistribution

def read_exfor(filepath: str) -> "ExforAngularDistribution":
    """
    Read an EXFOR angular distribution from a JSON file.

    This is the primary function for loading EXFOR data in kika. It supports
    both the standardized v1.0 schema and legacy formats.

    Parameters:
        filepath: Path to the JSON file

    Returns:
        ExforAngularDistribution object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid

    Example:
        >>> from kika.exfor import read_exfor
        >>> exfor = read_exfor('/path/to/27673002.json')
        >>> print(exfor.label)
        Gkatis et al. (2025)
        >>> print(exfor.energies())
        [1.0098, 1.0202, ...]
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXFOR file not found: {filepath}")

    return ExforAngularDistribution.from_json(filepath)


def read_all_exfor(
    target: Union[str, int] = None,
    mt: int = None,
    source: str = "database",
    directory: str = None,
    pattern: str = "*.json",
    group_by_energy: bool = True,
    db_path: str = None,
    projectile: str = "n",
    energy_range: Tuple[float, float] = None,
    supplementary_json_files: List[str] = None,
    # Deprecated parameter for backwards compatibility
    target_zaid: int = None,
) -> Union[Dict[float, List], Dict[str, "ExforAngularDistribution"]]:
    """
    Read EXFOR angular distribution data.

    This is the main function for loading EXFOR experimental data. By default,
    it loads from the X4Pro database (189k+ datasets). You can also load from
    your own curated JSON files.

    Parameters
    ----------
    target : str or int, optional
        Target isotope. Accepts multiple formats:
        - "Fe56" (symbol + mass)
        - 26056 (ZAID)
        - "Fe-56" (database format)
        - "26-FE-56" (EXFOR notation)
    mt : int, optional
        ENDF MT number (e.g., 2 for elastic scattering)
    source : str, optional
        Data source (default: "database"):
        - "database": Load from X4Pro database
        - "json": Load from JSON files only
        - "auto": Database with JSON fallback
    directory : str, optional
        Path to JSON files (required if source="json")
    group_by_energy : bool, optional
        If True (default), group by energy. If False, return dict by ID.
    db_path : str, optional
        Path to X4Pro database (uses default if None)
    projectile : str, optional
        Projectile (default: "n" for neutrons)
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV
    supplementary_json_files : List[str], optional
        List of additional JSON file paths to load. Useful for loading
        experiments not yet in the database (e.g., recent publications).
        These are loaded after the main source and deduplicated by ID.

    Returns
    -------
    Dict[float, List[ExforAngularDistribution]]
        Dictionary mapping energy (MeV) to list of datasets.
        If group_by_energy=False: Dict mapping ID to dataset.

    Examples
    --------
    >>> # Load Fe-56 elastic scattering data
    >>> data = read_all_exfor(target="Fe56", mt=2)

    >>> # Load using ZAID
    >>> data = read_all_exfor(target=26056, mt=2)

    >>> # Load only from your curated JSON files
    >>> data = read_all_exfor(source="json", directory="/path/to/data/")

    >>> # Load with energy filter
    >>> data = read_all_exfor(target="Fe56", energy_range=(1.0, 3.0))

    >>> # Load from database plus a supplementary JSON file
    >>> data = read_all_exfor(
    ...     target="Fe56", mt=2,
    ...     supplementary_json_files=["C:/EXFOR/27673002.json"]
    ... )
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    source = source.lower()
    valid_sources = {"json", "database", "auto", "both"}
    if source not in valid_sources:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {valid_sources}")

    # Handle backwards compatibility: target_zaid -> target
    if target_zaid is not None and target is None:
        target = target_zaid

    all_datasets: List[ExforAngularDistribution] = []
    loaded_ids: set = set()

    # Load from database if requested
    if source in ("database", "auto", "both"):
        try:
            db_datasets = _load_from_database(
                db_path=db_path,
                target=target,
                projectile=projectile,
                mt=mt,
                energy_range=energy_range,
            )
            for ds in db_datasets:
                ds_id = f"{ds.entry}{ds.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(ds)
                    loaded_ids.add(ds_id)
        except FileNotFoundError as e:
            if source == "database":
                raise
            # For auto/both, continue to JSON fallback
            print(f"Warning: Database not available, falling back to JSON: {e}")
        except Exception as e:
            if source == "database":
                raise
            print(f"Warning: Database query failed: {e}")

    # Load from JSON files if requested
    if source in ("json", "auto", "both"):
        if directory is not None:
            try:
                json_datasets = _load_from_json_files(directory, pattern)
                for ds in json_datasets:
                    ds_id = f"{ds.entry}{ds.subentry}"
                    if ds_id not in loaded_ids:
                        all_datasets.append(ds)
                        loaded_ids.add(ds_id)
            except FileNotFoundError as e:
                if source == "json":
                    raise
                # For auto/both, database data is sufficient
                if not all_datasets:
                    raise FileNotFoundError(
                        f"No data available. Database not found and {e}"
                    )

    # Load supplementary JSON files (for experiments not in database)
    if supplementary_json_files:
        for json_file in supplementary_json_files:
            if not os.path.exists(json_file):
                print(f"Warning: Supplementary file not found: {json_file}")
                continue
            try:
                exfor = read_exfor(json_file)
                ds_id = f"{exfor.entry}{exfor.subentry}"
                if ds_id not in loaded_ids:
                    all_datasets.append(exfor)
                    loaded_ids.add(ds_id)
            except Exception as e:
                print(f"Warning: Could not load supplementary file {json_file}: {e}")

    if not all_datasets:
        raise FileNotFoundError("No EXFOR data found from any source")

    # Organize results
    if not group_by_energy:
        return {f"{ds.entry}{ds.subentry}": ds for ds in all_datasets}

    # Group by energy
    energy_data: Dict[float, List[ExforAngularDistribution]] = {}
    for exfor in all_datasets:
        energies_mev = exfor.energies(unit='MeV')
        for energy in energies_mev:
            key = _resolve_energy_key(energy, energy_data)
            if key not in energy_data:
                energy_data[key] = []
            if exfor not in energy_data[key]:
                energy_data[key].append(exfor)

    return energy_data


def _load_from_database(
    db_path: str = None,
    target: Union[str, int] = None,
    projectile: str = "n",
    mt: int = None,
    energy_range: Tuple[float, float] = None,
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from X4Pro database.

    Parameters
    ----------
    db_path : str, optional
        Path to database file
    target : str or int, optional
        Target (accepts "Fe56", 26056, "Fe-56", etc.)
    projectile : str, optional
        Projectile (default: "n")
    mt : int, optional
        ENDF MT number
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from database
    """
    from kika.exfor.database import X4ProDatabase

    with X4ProDatabase(db_path) as db:
        # Normalize target to database format
        target_pattern = db._normalize_target(target) if target else None

        return db.query_angular_distributions(
            target=target_pattern,
            projectile=projectile,
            mt=mt,
            energy_range=energy_range,
        )


def _load_from_json_files(
    directory: str,
    pattern: str = "*.json",
) -> List["ExforAngularDistribution"]:
    """
    Load EXFOR data from JSON files in a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing JSON files
    pattern : str, optional
        Glob pattern for file matching

    Returns
    -------
    List[ExforAngularDistribution]
        List of datasets from JSON files
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    search_dir = _find_data_directory(directory)
    if search_dir is None:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    json_files = glob.glob(os.path.join(search_dir, pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {search_dir}")

    datasets = []
    for json_file in json_files:
        try:
            exfor = read_exfor(json_file)
            datasets.append(exfor)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return datasets


def _find_data_directory(directory: str) -> Optional[str]:
    """
    Find directory containing JSON files.

    Checks the given directory, then looks for 'data' subdirectory.

    Parameters:
        directory: Starting directory path

    Returns:
        Path to directory with JSON files, or None if not found
    """
    # Check if directory exists and has JSON files
    if os.path.isdir(directory):
        if glob.glob(os.path.join(directory, "*.json")):
            return directory
        # Try 'data' subdirectory
        data_subdir = os.path.join(directory, "data")
        if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
            return data_subdir

    # Try parent directory with 'data' subdirectory
    parent_dir = os.path.dirname(directory.rstrip("/\\"))
    data_subdir = os.path.join(parent_dir, "data")
    if os.path.isdir(data_subdir) and glob.glob(os.path.join(data_subdir, "*.json")):
        return data_subdir

    return None


def _resolve_energy_key(energy: float, energy_dict: Dict[float, List]) -> float:
    """
    Find existing energy key within tolerance, or return the new energy.

    Parameters:
        energy: Energy value to match
        energy_dict: Existing dictionary with energy keys

    Returns:
        Existing key if within tolerance, otherwise the input energy
    """
    for existing in energy_dict.keys():
        if math.isclose(existing, energy, rel_tol=0.0, abs_tol=ENERGY_MATCH_ABS_TOL):
            return existing
    return energy
