"""
EXFOR JSON file reading functions.

This module provides the main I/O functions for loading EXFOR data from
JSON files and organizing datasets by energy.
"""

import os
import glob
import math
from typing import Dict, List, Optional, Union

from kika.exfor._constants import ENERGY_MATCH_ABS_TOL


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
    directory: str,
    pattern: str = "*.json",
    group_by_energy: bool = True,
) -> Union[Dict[float, List], Dict[str, "ExforAngularDistribution"]]:
    """
    Read all EXFOR JSON files from a directory and organize by energy.

    Parameters:
        directory: Path to directory containing EXFOR JSON files
        pattern: Glob pattern for file matching (default: "*.json")
        group_by_energy: If True, group by energy; if False, return flat dict by filename

    Returns:
        Dictionary mapping energy (MeV) to list of ExforAngularDistribution objects.
        If group_by_energy=False, returns dict mapping filename to ExforAngularDistribution.

    Example:
        >>> from kika.exfor import read_all_exfor
        >>> energy_dict = read_all_exfor('/path/to/data/')
        >>> energies = sorted(energy_dict.keys())
        >>> for energy in energies[:3]:
        ...     print(f"E={energy} MeV: {len(energy_dict[energy])} datasets")
    """
    from kika.exfor.angular_distribution import ExforAngularDistribution

    # Find directory with JSON files
    search_dir = _find_data_directory(directory)
    if search_dir is None:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    json_files = glob.glob(os.path.join(search_dir, pattern))
    if not json_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {search_dir}")

    if not group_by_energy:
        # Return dict by filename
        result = {}
        for json_file in json_files:
            try:
                exfor = read_exfor(json_file)
                filename = os.path.basename(json_file)
                result[filename] = exfor
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        return result

    # Group by energy
    energy_data: Dict[float, List[ExforAngularDistribution]] = {}

    for json_file in json_files:
        try:
            exfor = read_exfor(json_file)

            # Get energies in MeV
            energies_mev = exfor.energies(unit='MeV')

            for energy in energies_mev:
                key = _resolve_energy_key(energy, energy_data)
                if key not in energy_data:
                    energy_data[key] = []
                # Only add once per file (we store the whole object)
                if exfor not in energy_data[key]:
                    energy_data[key].append(exfor)

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    return energy_data


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
