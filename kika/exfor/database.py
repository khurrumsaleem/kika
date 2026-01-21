"""
X4Pro EXFOR Database Interface.

This module provides the X4ProDatabase class for accessing EXFOR data
directly from the X4Pro SQLite database (full 2025 version with JSON schema).

The full database stores measurement data in JSON format in the `x4pro_x5z.jx5z`
column, containing:
- x4data: Raw measurement arrays (energy, angle, cross section, uncertainties)
- c5data: Pre-computed data arrays
- Metadata: author, year, target, projectile, reaction codes

Usage:
    >>> from kika.exfor.database import X4ProDatabase
    >>> db = X4ProDatabase()  # Uses KIKA_X4PRO_DB_PATH env var or default path
    >>> datasets = db.query_angular_distributions(target_zaid=26056, projectile="N")
    >>> print(f"Found {len(datasets)} angular distribution datasets")
"""

import json
import os
import sqlite3
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kika.exfor._constants import (
    FRAME_LAB,
    FRAME_CM,
    DB_DEFAULT_PATH,
    DB_UNIT_MAPPINGS,
    DB_FAMILY_MAPPINGS,
)
from kika.exfor.config import get_tof_metadata_path
from kika._constants import ATOMIC_NUMBER_TO_SYMBOL, SYMBOL_TO_ATOMIC_NUMBER
from kika._utils import zaid_to_symbol, symbol_to_zaid
from kika.exfor.angular_distribution import ExforAngularDistribution
from kika.exfor.exfor_entry import ExforEntry

# Module-level cache for TOF metadata
_tof_metadata_cache: Optional[Dict[str, Any]] = None


def _load_tof_metadata(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load TOF metadata from the configuration file.

    The metadata file contains flight path and time resolution parameters
    for experiments, which supplements the database that lacks this info.

    Parameters
    ----------
    force_reload : bool
        If True, reload from file even if cached

    Returns
    -------
    Dict[str, Any]
        TOF metadata with 'default' and 'experiments' keys
    """
    global _tof_metadata_cache

    if _tof_metadata_cache is not None and not force_reload:
        return _tof_metadata_cache

    metadata_path = get_tof_metadata_path()

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                _tof_metadata_cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Fall back to defaults if file is invalid
            _tof_metadata_cache = {
                "default": {"flight_path_m": 27.037, "time_resolution_ns": 10.0},
                "experiments": {},
            }
    else:
        # No metadata file - use defaults
        _tof_metadata_cache = {
            "default": {"flight_path_m": 27.037, "time_resolution_ns": 10.0},
            "experiments": {},
        }

    return _tof_metadata_cache


def _get_tof_params_for_experiment(dataset_id: str) -> Dict[str, Any]:
    """
    Get TOF parameters for a specific experiment.

    Looks up the experiment in the metadata file. If not found,
    returns the default values.

    Parameters
    ----------
    dataset_id : str
        EXFOR dataset ID (e.g., "10037024")

    Returns
    -------
    Dict[str, Any]
        Dictionary with 'flight_path_m' and 'time_resolution_ns' keys
    """
    metadata = _load_tof_metadata()
    defaults = metadata.get("default", {"flight_path_m": 27.037, "time_resolution_ns": 10.0})
    experiments = metadata.get("experiments", {})

    if dataset_id in experiments:
        exp_data = experiments[dataset_id]
        return {
            "flight_path_m": exp_data.get("flight_path_m", defaults["flight_path_m"]),
            "time_resolution_ns": exp_data.get("time_resolution_ns", defaults["time_resolution_ns"]),
        }

    return defaults.copy()

@dataclass
class X4ProDataset:
    """
    Raw dataset from X4Pro database before conversion to ExforAngularDistribution.

    This intermediate representation holds the parsed JSON data from the database
    before it is converted to the full ExforAngularDistribution object.
    """

    dataset_id: str
    year: int
    author: str
    target: str
    projectile: str
    mf: int
    mt: int
    quant: str
    ndat: int
    reacode: str

    # Parsed data arrays
    energies_ev: np.ndarray = field(default_factory=lambda: np.array([]))
    angles_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    cross_sections: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainties: np.ndarray = field(default_factory=lambda: np.array([]))

    # Units as read from database
    energy_unit: str = "EV"
    angle_unit: str = "ADEG"
    xs_unit: str = "B/SR"

    # Frame information
    angle_frame: str = FRAME_LAB

    # Correction information
    is_corrected: bool = False
    correction_notes: List[str] = field(default_factory=list)

    # Raw JSON for debugging/verification
    raw_json: Optional[Dict[str, Any]] = None


def _zaid_to_target_pattern(zaid: int) -> str:
    """
    Convert ZAID to database target pattern (e.g., 26056 -> "Fe-56").

    Uses kika._utils.zaid_to_symbol internally.
    """
    symbol = zaid_to_symbol(zaid)  # Returns "Fe56" or "Fe" for natural
    # Convert "Fe56" to "Fe-56" format used by database
    match = re.match(r"([A-Za-z]+)(\d*)", symbol)
    if match:
        elem = match.group(1)
        mass = match.group(2) or "0"
        return f"{elem}-{mass}"
    return ""


def _parse_target_from_db(target_str: str) -> Tuple[str, int]:
    """
    Parse target from database format (e.g., "Fe-56" -> ("Fe56", 26056)).

    Uses kika._utils.symbol_to_zaid internally.
    """
    # Try database format first: "Fe-56" -> "Fe56"
    match = re.match(r"([A-Za-z]+)-(\d+)", target_str)
    if match:
        symbol = match.group(1).capitalize()
        a = match.group(2)
        target = f"{symbol}{a}" if a != "0" else symbol
        try:
            zaid = symbol_to_zaid(target)
            return (target, zaid)
        except ValueError:
            pass

    # Try EXFOR format: "26-FE-56"
    match = re.match(r"(\d+)-([A-Z]+)-(\d+)", target_str.upper())
    if match:
        z = int(match.group(1))
        symbol = match.group(2).capitalize()
        a = int(match.group(3))
        target = f"{symbol}{a}" if a > 0 else f"{symbol}0"
        zaid = z * 1000 + a
        return (target, zaid)

    return ("Unknown", 0)


def _parse_x4data_json(jx5z: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the x4data array from the JSON structure.

    The x4data array contains measurement variables:
    - ivar: Variable index
    - cvar: Variable category ('y' for dependent, 'x1', 'x2' for independent)
    - fam: Family code ('Data', 'EN', 'ANG', 'COS', 'dData', etc.)
    - dat0: Data array
    - units: Unit string

    Parameters
    ----------
    jx5z : Dict[str, Any]
        Parsed JSON from jx5z column

    Returns
    -------
    Dict[str, Any]
        Extracted data with keys: 'energies', 'angles', 'values', 'uncertainties',
        'energy_unit', 'angle_unit', 'xs_unit', 'angle_type'
    """
    x4data = jx5z.get("x4data", [])

    result = {
        "energies": [],
        "angles": [],
        "values": [],
        "uncertainties": [],
        "energy_unit": "EV",
        "angle_unit": "ADEG",
        "xs_unit": "B/SR",
        "angle_type": "ANG",  # 'ANG' or 'COS'
    }

    for var in x4data:
        fam = var.get("fam", "")
        cvar = var.get("cvar", "")
        dat0 = var.get("dat0", [])
        units = var.get("units", "")

        if fam == "Data" and cvar == "y":
            # Cross section values
            result["values"] = dat0
            result["xs_unit"] = units
        elif fam == "EN":
            # Energy values
            result["energies"] = dat0
            result["energy_unit"] = units
        elif fam == "ANG":
            # Angle in degrees
            result["angles"] = dat0
            result["angle_unit"] = units
            result["angle_type"] = "ANG"
        elif fam == "COS":
            # Angle as cosine
            result["angles"] = dat0
            result["angle_unit"] = units
            result["angle_type"] = "COS"
        elif cvar == "dy" or fam in ("dData", "DATA-ERR"):
            # Uncertainty values
            result["uncertainties"] = dat0

    return result


def _parse_c5data_json(jx5z: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the c5data (corrected) dict from the JSON structure.

    The c5data dict contains pre-computed values with corrections applied
    (when applicable) and standardized units. This is the preferred data
    source as it includes:
    - Decay data corrections (gamma line intensities from ENSDF)
    - Monitor cross section renormalization
    - Standardized units (EV, ADEG, B/SR)

    Parameters
    ----------
    jx5z : Dict[str, Any]
        Parsed JSON from jx5z column

    Returns
    -------
    Dict[str, Any]
        Extracted data with keys: 'energies', 'angles', 'values', 'uncertainties',
        'energy_unit', 'angle_unit', 'xs_unit', 'angle_type', 'is_corrected',
        'correction_notes'
    """
    c5data = jx5z.get("c5data", {})

    result = {
        "energies": [],
        "angles": [],
        "values": [],
        "uncertainties": [],
        "energy_unit": "EV",
        "angle_unit": "ADEG",
        "xs_unit": "B/SR",
        "angle_type": "ANG",
        "is_corrected": False,
        "correction_notes": [],
    }

    if not isinstance(c5data, dict):
        return result

    # Extract y (cross section)
    if "y" in c5data:
        y_data = c5data["y"]
        if isinstance(y_data, dict):
            result["values"] = y_data.get("y", [])
            result["uncertainties"] = y_data.get("dy", [])
            result["xs_unit"] = y_data.get("units", "B/SR")

    # Extract x1 (energy)
    if "x1" in c5data:
        x1_data = c5data["x1"]
        if isinstance(x1_data, dict):
            # Check family is energy (EN)
            if x1_data.get("fam") == "EN":
                result["energies"] = x1_data.get("x1", [])
                result["energy_unit"] = x1_data.get("units", "EV")

    # Extract x2 (angle or cosine)
    if "x2" in c5data:
        x2_data = c5data["x2"]
        if isinstance(x2_data, dict):
            result["angles"] = x2_data.get("x2", [])
            fam = x2_data.get("fam", "")
            if fam == "COS":
                result["angle_type"] = "COS"
                result["angle_unit"] = "NO-DIM"
            else:
                result["angle_type"] = "ANG"
                result["angle_unit"] = x2_data.get("units", "ADEG")

    # Check if corrections were applied
    auto_corr_notes = jx5z.get("autoCorrNotes", [])
    if auto_corr_notes:
        result["is_corrected"] = True
        result["correction_notes"] = auto_corr_notes if isinstance(auto_corr_notes, list) else [auto_corr_notes]

    return result


def _convert_units(
    values: np.ndarray, from_unit: str, target_unit: str, unit_type: str
) -> np.ndarray:
    """
    Convert values between units.

    Parameters
    ----------
    values : np.ndarray
        Input values
    from_unit : str
        Source unit (e.g., 'EV', 'MEV', 'B/SR', 'MB/SR')
    target_unit : str
        Target unit
    unit_type : str
        Type of unit: 'energy', 'cross_section'

    Returns
    -------
    np.ndarray
        Converted values
    """
    from_unit = from_unit.upper()
    target_unit = target_unit.upper()

    if from_unit == target_unit:
        return values

    mappings = DB_UNIT_MAPPINGS.get(unit_type, {})

    from_factor = mappings.get(from_unit, 1.0)
    to_factor = mappings.get(target_unit, 1.0)

    return values * (from_factor / to_factor)


class X4ProDatabase:
    """
    Interface to X4Pro SQLite database (full 2025 version).

    This class provides methods for querying angular distribution data from the
    X4Pro database and converting it to ExforAngularDistribution objects.

    Parameters
    ----------
    db_path : str, optional
        Path to X4Pro SQLite database. If not provided, uses the
        KIKA_X4PRO_DB_PATH environment variable.

    Attributes
    ----------
    db_path : str
        Path to the database file
    _conn : sqlite3.Connection
        Database connection (lazy-loaded)

    Examples
    --------
    >>> db = X4ProDatabase()
    >>> datasets = db.query_angular_distributions(target_zaid=26056)
    >>> print(f"Found {len(datasets)} datasets for Fe-56")
    """

    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        from kika.exfor.config import get_db_path
        self.db_path = get_db_path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            if self.db_path is None:
                raise FileNotFoundError(
                    "X4Pro database path not configured.\n"
                    "Set KIKA_X4PRO_DB_PATH environment variable or provide db_path parameter."
                )
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"X4Pro database not found at: {self.db_path}\n"
                    f"Set KIKA_X4PRO_DB_PATH environment variable or provide db_path."
                )
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def query_dataset_ids(
        self,
        target: str = None,
        target_zaid: int = None,
        projectile: str = "N",
        quantity: str = "DA",
        mf: int = None,
        mt: int = None,
        energy_min_mev: float = None,
        energy_max_mev: float = None,
        year_min: int = None,
        year_max: int = None,
        author: str = None,
    ) -> List[str]:
        """
        Query dataset IDs matching the given criteria.

        Parameters
        ----------
        target : str, optional
            Target in EXFOR notation (e.g., "26-FE-56")
        target_zaid : int, optional
            Target ZAID (e.g., 26056 for Fe-56)
        projectile : str, optional
            Projectile (default: "N" for neutrons)
        quantity : str, optional
            Quantity type (default: "DA" for angular distribution)
        mf : int, optional
            ENDF MF number (4 for angular distributions)
        mt : int, optional
            ENDF MT number (2 for elastic scattering)
        energy_min_mev : float, optional
            Minimum energy in MeV
        energy_max_mev : float, optional
            Maximum energy in MeV
        year_min : int, optional
            Minimum publication year
        year_max : int, optional
            Maximum publication year
        author : str, optional
            Author name (partial match)

        Returns
        -------
        List[str]
            List of DatasetID strings
        """
        conn = self._get_connection()

        # Build query
        conditions = []
        params = []

        # Target filtering
        if target:
            conditions.append("Targ1 LIKE ?")
            params.append(f"%{target}%")
        elif target_zaid:
            # Convert ZAID to database target format (e.g., 26056 -> "Fe-56")
            target_pattern = _zaid_to_target_pattern(target_zaid)
            if target_pattern:
                conditions.append("Targ1 = ?")
                params.append(target_pattern)

        # Projectile (database uses lowercase for common particles: n, p, d, a, g)
        if projectile:
            proj_lower = projectile.lower()
            # Common particles are stored lowercase; heavier ions may be mixed case
            conditions.append("(Proj = ? OR Proj = ?)")
            params.extend([proj_lower, projectile.upper()])

        # Quantity (angular distribution)
        if quantity:
            conditions.append("quant1 LIKE ?")
            params.append(f"%{quantity}%")

        # MF/MT numbers
        if mf is not None:
            conditions.append("MF = ?")
            params.append(mf)
        if mt is not None:
            conditions.append("MT = ?")
            params.append(mt)

        # Year range
        if year_min is not None:
            conditions.append("year1 >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("year1 <= ?")
            params.append(year_max)

        # Author
        if author:
            conditions.append("author1 LIKE ?")
            params.append(f"%{author}%")

        # Build final query
        query = "SELECT DatasetID FROM x4pro_ds"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]

    def get_dataset_json(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the raw JSON data for a dataset.

        Parameters
        ----------
        dataset_id : str
            Dataset ID (e.g., "10012002")

        Returns
        -------
        Dict[str, Any] or None
            Parsed JSON from jx5z column, or None if not found
        """
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT jx5z FROM x4pro_x5z WHERE DatasetID = ?", (dataset_id,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None

    def get_dataset_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a dataset from x4pro_ds table.

        Parameters
        ----------
        dataset_id : str
            Dataset ID

        Returns
        -------
        Dict[str, Any] or None
            Dataset metadata
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT DatasetID, year1, author1, Targ1, Proj, MF, MT, ndat, quant1, reacode
            FROM x4pro_ds
            WHERE DatasetID = ?
            """,
            (dataset_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "dataset_id": row["DatasetID"],
            "year": row["year1"],
            "author": row["author1"],
            "target": row["Targ1"],
            "projectile": row["Proj"],
            "mf": row["MF"],
            "mt": row["MT"],
            "ndat": row["ndat"],
            "quant": row["quant1"],
            "reacode": row["reacode"],
        }

    def parse_dataset(self, dataset_id: str) -> Optional[X4ProDataset]:
        """
        Parse a dataset from the database into X4ProDataset.

        This method prefers c5data (corrected data) over x4data (raw data)
        because c5data contains:
        - Pre-applied corrections (decay data, monitor renormalization)
        - Standardized units (EV, ADEG, B/SR)
        - Cleaner structure

        Parameters
        ----------
        dataset_id : str
            Dataset ID

        Returns
        -------
        X4ProDataset or None
            Parsed dataset, or None if not found
        """
        # Get metadata
        metadata = self.get_dataset_metadata(dataset_id)
        if metadata is None:
            return None

        # Get JSON data
        jx5z = self.get_dataset_json(dataset_id)
        if jx5z is None:
            return None

        # Try c5data first (contains corrected values with standardized units)
        parsed = _parse_c5data_json(jx5z)
        is_corrected = parsed["is_corrected"]
        correction_notes = parsed["correction_notes"]

        # Convert to numpy arrays
        energies = np.array(parsed["energies"], dtype=float) if parsed["energies"] else np.array([])
        angles = np.array(parsed["angles"], dtype=float) if parsed["angles"] else np.array([])
        values = np.array(parsed["values"], dtype=float) if parsed["values"] else np.array([])
        uncertainties = np.array(parsed["uncertainties"], dtype=float) if parsed["uncertainties"] else np.array([])
        energy_unit = parsed["energy_unit"]
        xs_unit = parsed["xs_unit"]
        angle_type = parsed["angle_type"]

        # Fallback to x4data if c5data is empty/incomplete
        if len(values) == 0:
            x4_parsed = _parse_x4data_json(jx5z)
            if x4_parsed["values"]:
                values = np.array(x4_parsed["values"], dtype=float)
                xs_unit = x4_parsed["xs_unit"]
            if x4_parsed["uncertainties"]:
                uncertainties = np.array(x4_parsed["uncertainties"], dtype=float)

        if len(energies) == 0:
            x4_parsed = _parse_x4data_json(jx5z)
            if x4_parsed["energies"]:
                energies = np.array(x4_parsed["energies"], dtype=float)
                energy_unit = x4_parsed["energy_unit"]

        if len(angles) == 0:
            x4_parsed = _parse_x4data_json(jx5z)
            if x4_parsed["angles"]:
                angles = np.array(x4_parsed["angles"], dtype=float)
                angle_type = x4_parsed["angle_type"]

        # Ensure uncertainties array is initialized
        if len(uncertainties) == 0 and len(values) > 0:
            uncertainties = np.zeros_like(values)

        # Handle angle type (convert cosine to degrees if needed)
        if angle_type == "COS":
            angles = np.degrees(np.arccos(np.clip(angles, -1.0, 1.0)))
            angle_unit = "ADEG"
        else:
            angle_unit = parsed["angle_unit"]

        # Determine frame from reacode
        reacode = metadata.get("reacode", "")
        angle_frame = FRAME_CM if ",DA/DA,," in reacode or angle_type == "COS" else FRAME_LAB

        return X4ProDataset(
            dataset_id=dataset_id,
            year=metadata["year"],
            author=metadata["author"],
            target=metadata["target"],
            projectile=metadata["projectile"],
            mf=metadata["mf"],
            mt=metadata["mt"],
            quant=metadata["quant"],
            ndat=metadata["ndat"],
            reacode=metadata["reacode"],
            energies_ev=energies,
            angles_deg=angles,
            cross_sections=values,
            uncertainties=uncertainties,
            energy_unit=energy_unit,
            angle_unit=angle_unit,
            xs_unit=xs_unit,
            angle_frame=angle_frame,
            is_corrected=is_corrected,
            correction_notes=correction_notes,
            raw_json=jx5z,
        )

    def query_angular_distributions(
        self,
        target: str = None,
        target_zaid: int = None,
        projectile: str = "N",
        process: str = None,
        energy_range: Tuple[float, float] = None,
        mt: int = None,
        convert_to_objects: bool = True,
    ) -> Union[List["ExforAngularDistribution"], List[X4ProDataset]]:
        """
        Query angular distribution datasets from the database.

        This is the main method for retrieving EXFOR angular distribution data.
        It queries datasets matching the criteria and optionally converts them
        to ExforAngularDistribution objects.

        Parameters
        ----------
        target : str, optional
            Target in EXFOR notation (e.g., "26-FE-56")
        target_zaid : int, optional
            Target ZAID (e.g., 26056 for Fe-56). Alternative to target.
        projectile : str, optional
            Projectile (default: "N" for neutrons)
        process : str, optional
            Reaction process (e.g., "EL" for elastic). If provided, filters by MT.
        energy_range : Tuple[float, float], optional
            Energy range (min, max) in MeV. Filters datasets with data in range.
        mt : int, optional
            ENDF MT number (e.g., 2 for elastic scattering)
        convert_to_objects : bool, optional
            If True (default), convert to ExforAngularDistribution objects.
            If False, return X4ProDataset objects.

        Returns
        -------
        List[ExforAngularDistribution] or List[X4ProDataset]
            List of angular distribution datasets

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> # Get all elastic scattering data for Fe-56
        >>> datasets = db.query_angular_distributions(target_zaid=26056, mt=2)
        >>> # Get data in specific energy range
        >>> datasets = db.query_angular_distributions(
        ...     target_zaid=26056,
        ...     energy_range=(1.0, 3.0)
        ... )
        """
        # Determine MT from process if not specified
        if mt is None and process:
            process_to_mt = {"EL": 2, "INL": 4, "N,N'": 4, "TOT": 1}
            mt = process_to_mt.get(process.upper())

        # Query dataset IDs
        dataset_ids = self.query_dataset_ids(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            quantity="DA",
            mf=4,  # MF=4 for angular distributions
            mt=mt,
        )

        # Parse datasets
        parsed_datasets = []
        for ds_id in dataset_ids:
            dataset = self.parse_dataset(ds_id)
            if dataset is None:
                continue

            # Apply energy filter if specified
            if energy_range is not None:
                e_min, e_max = energy_range
                # Convert energy range to eV for comparison
                e_min_ev = e_min * 1e6
                e_max_ev = e_max * 1e6

                # Check if dataset has data in range
                ds_energies = dataset.energies_ev
                if dataset.energy_unit.upper() == "MEV":
                    ds_energies = ds_energies * 1e6
                elif dataset.energy_unit.upper() == "KEV":
                    ds_energies = ds_energies * 1e3

                if len(ds_energies) == 0:
                    continue
                if np.max(ds_energies) < e_min_ev or np.min(ds_energies) > e_max_ev:
                    continue

            parsed_datasets.append(dataset)

        if not convert_to_objects:
            return parsed_datasets

        # Convert to ExforAngularDistribution objects
        return [self._convert_to_exfor_object(ds) for ds in parsed_datasets]

    def _convert_to_exfor_object(
        self, dataset: X4ProDataset
    ) -> "ExforEntry":
        """
        Convert X4ProDataset to appropriate ExforEntry subclass.

        The object type is determined by the quantity field (quant):
        - "DA" -> ExforAngularDistribution
        - Other types -> NotImplementedError (for now)

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforEntry
            Appropriate ExforEntry subclass based on data type

        Raises
        ------
        NotImplementedError
            If the quantity type is not yet supported
        """
        quantity = dataset.quant.upper() if dataset.quant else ""

        # Check for angular distribution (DA)
        if "DA" in quantity:
            return self._convert_to_angular_distribution(dataset)

        # Future: add other types here
        # if "SIG" in quantity:
        #     return self._convert_to_cross_section(dataset)

        raise NotImplementedError(
            f"Quantity type '{quantity}' is not yet supported. "
            f"Currently only angular distributions (DA) are implemented."
        )

    def _convert_to_angular_distribution(
        self, dataset: X4ProDataset
    ) -> "ExforAngularDistribution":
        """
        Convert X4ProDataset to ExforAngularDistribution.

        Parameters
        ----------
        dataset : X4ProDataset
            Parsed dataset from database

        Returns
        -------
        ExforAngularDistribution
            Full ExforAngularDistribution object
        """
        from kika.exfor.angular_distribution import ExforAngularDistribution

        # Parse target
        target_name, target_zaid = _parse_target_from_db(dataset.target)

        # Convert energies to MeV
        energies_mev = _convert_units(
            dataset.energies_ev,
            dataset.energy_unit,
            "MEV",
            "energy",
        )

        # Convert cross sections to b/sr
        xs_bsr = _convert_units(
            dataset.cross_sections,
            dataset.xs_unit,
            "B/SR",
            "cross_section",
        )
        unc_bsr = _convert_units(
            dataset.uncertainties,
            dataset.xs_unit,
            "B/SR",
            "cross_section",
        )

        # Group data by energy
        unique_energies = np.unique(energies_mev)
        data_blocks = []

        for energy in unique_energies:
            mask = np.isclose(energies_mev, energy, rtol=1e-6)
            block_angles = dataset.angles_deg[mask]
            block_xs = xs_bsr[mask]
            block_unc = unc_bsr[mask]

            # Sort by angle
            sort_idx = np.argsort(block_angles)
            block_angles = block_angles[sort_idx]
            block_xs = block_xs[sort_idx]
            block_unc = block_unc[sort_idx]

            data_points = []
            for i in range(len(block_angles)):
                data_points.append({
                    "angle": float(block_angles[i]),
                    "cross_section": float(block_xs[i]),
                    "uncertainty_stat": float(block_unc[i]),
                })

            data_blocks.append({
                "value": float(energy),
                "data": data_points,
            })

        # Extract entry/subentry from dataset_id
        entry = dataset.dataset_id[:5]
        subentry = dataset.dataset_id[5:]

        # Build citation
        author_parts = dataset.author.split(".")
        surname = author_parts[-1] if author_parts else dataset.author

        citation = {
            "authors": [dataset.author],
            "year": dataset.year,
            "reference": f"EXFOR {dataset.dataset_id}",
        }

        # Build reaction
        reaction = {
            "target": target_name,
            "target_zaid": target_zaid,
            "projectile": dataset.projectile.lower(),
            "process": "EL" if dataset.mt == 2 else f"MT{dataset.mt}",
            "notation": dataset.reacode,
        }

        # Get TOF parameters from metadata file
        tof_params = _get_tof_params_for_experiment(dataset.dataset_id)
        method = {
            "type": "TOF",
            "energy_resolution_input": {
                "distance": {
                    "value": tof_params["flight_path_m"],
                    "unit": "m",
                },
                "time_resolution": {
                    "value": tof_params["time_resolution_ns"],
                    "unit": "ns",
                },
            },
        }

        return ExforAngularDistribution(
            entry=entry,
            subentry=subentry,
            quantity="DA",
            citation=citation,
            reaction=reaction,
            facility={},
            method=method,
            angle_frame=dataset.angle_frame,
            units={"energy": "MeV", "angle": "deg", "cross_section": "b/sr"},
            _data_blocks=data_blocks,
        )

    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns
        -------
        Dict[str, int]
            Statistics including total datasets, angular distributions, etc.
        """
        conn = self._get_connection()

        stats = {}

        # Total datasets
        cursor = conn.execute("SELECT COUNT(*) FROM x4pro_x5z")
        stats["total_datasets"] = cursor.fetchone()[0]

        # Total metadata entries
        cursor = conn.execute("SELECT COUNT(*) FROM x4pro_ds")
        stats["total_metadata"] = cursor.fetchone()[0]

        # Angular distributions (DA)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM x4pro_ds WHERE quant1 LIKE '%DA%'"
        )
        stats["angular_distributions"] = cursor.fetchone()[0]

        # Elastic scattering
        cursor = conn.execute(
            "SELECT COUNT(*) FROM x4pro_ds WHERE quant1 LIKE '%DA%' AND MT = 2"
        )
        stats["elastic_scattering"] = cursor.fetchone()[0]

        return stats

    def list_targets(self, projectile: str = "n") -> List[str]:
        """
        List all unique targets with angular distribution data.

        Parameters
        ----------
        projectile : str, optional
            Filter by projectile (default: "n" for neutrons)

        Returns
        -------
        List[str]
            Sorted list of unique target strings (e.g., ["Fe-54", "Fe-56", "Fe-57"])

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> targets = db.list_targets()
        >>> print(targets[:5])
        ['Ag-0', 'Ag-107', 'Ag-109', 'Al-27', 'Am-241']
        """
        conn = self._get_connection()
        proj_lower = projectile.lower()

        cursor = conn.execute(
            """SELECT DISTINCT Targ1 FROM x4pro_ds
               WHERE (Proj = ? OR Proj = ?) AND quant1 LIKE '%DA%'""",
            (proj_lower, projectile.upper()),
        )
        targets = [row[0] for row in cursor.fetchall() if row[0]]
        return sorted(targets)

    def list_experiments(
        self,
        target: Union[str, int] = None,
        projectile: str = "n",
        mt: int = None,
    ) -> "pd.DataFrame":
        """
        List all experiments for a target with summary info.

        Parameters
        ----------
        target : str or int, optional
            Target isotope. Accepts multiple formats:
            - "Fe56" or "Fe-56" (symbol + mass)
            - 26056 (ZAID)
            - None to list all targets
        projectile : str, optional
            Projectile (default: "n" for neutrons)
        mt : int, optional
            ENDF MT number (e.g., 2 for elastic)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: dataset_id, author, year, energy_min, energy_max

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> experiments = db.list_experiments("Fe56", mt=2)
        >>> print(experiments)
           dataset_id    author  year  energy_min  energy_max
        0    10037024  Boschung  1971        0.01        0.01
        1    10571002    Kinney  1970        4.07        8.56
        """
        import pandas as pd

        conn = self._get_connection()

        # Build query conditions
        conditions = ["quant1 LIKE '%DA%'"]
        params = []

        # Handle target - accept multiple formats
        if target is not None:
            target_pattern = self._normalize_target(target)
            if target_pattern:
                conditions.append("Targ1 = ?")
                params.append(target_pattern)

        # Projectile
        proj_lower = projectile.lower()
        conditions.append("(Proj = ? OR Proj = ?)")
        params.extend([proj_lower, projectile.upper()])

        # MT number
        if mt is not None:
            conditions.append("MT = ?")
            params.append(mt)

        query = f"""
            SELECT DatasetID, author1, year1, Targ1, MT
            FROM x4pro_ds
            WHERE {' AND '.join(conditions)}
            ORDER BY year1, author1
        """

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        # Get energy ranges from JSON data
        results = []
        for row in rows:
            dataset_id = row[0]
            # Get energy range from parsed data
            jx5z = self.get_dataset_json(dataset_id)
            e_min, e_max = None, None
            if jx5z:
                parsed = _parse_x4data_json(jx5z)
                energies = None
                unit = None

                # Try x4data first
                if parsed["energies"]:
                    energies = np.array(parsed["energies"], dtype=float)
                    unit = parsed["energy_unit"].upper()

                # Fallback to c5data if x4data has no energies
                if energies is None or len(energies) == 0:
                    c5data = jx5z.get("c5data", {})
                    if isinstance(c5data, dict) and "x1" in c5data:
                        x1 = c5data["x1"]
                        if x1.get("fam") == "EN" and "x1" in x1:
                            c5_energies = x1.get("x1", [])
                            if c5_energies:
                                energies = np.array(c5_energies, dtype=float)
                                unit = x1.get("units", "EV").upper()

                if energies is not None and len(energies) > 0:
                    # Remove NaN values
                    energies = energies[~np.isnan(energies)]
                    if len(energies) > 0:
                        # Convert to MeV
                        if unit == "EV":
                            energies = energies / 1e6
                        elif unit == "KEV":
                            energies = energies / 1e3
                        e_min = float(np.min(energies))
                        e_max = float(np.max(energies))

            results.append({
                "dataset_id": dataset_id,
                "author": row[1],
                "year": row[2],
                "energy_min": e_min,
                "energy_max": e_max,
            })

        return pd.DataFrame(results)

    def load_experiment(self, dataset_id: str) -> "ExforEntry":
        """
        Load a specific experiment by its dataset ID.

        Returns the appropriate ExforEntry subclass based on data type:
        - Angular distributions (DA) -> ExforAngularDistribution
        - Other types -> NotImplementedError (for now)

        Parameters
        ----------
        dataset_id : str
            EXFOR dataset identifier (e.g., "10037024")

        Returns
        -------
        ExforEntry
            The loaded experiment data as appropriate subclass

        Raises
        ------
        ValueError
            If the dataset is not found
        NotImplementedError
            If the quantity type is not yet supported

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> exp = db.load_experiment("10037024")
        >>> print(exp.label)
        Boschung (1971)
        >>> isinstance(exp, ExforAngularDistribution)
        True
        """
        dataset = self.parse_dataset(dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset {dataset_id} not found in database")
        return self._convert_to_exfor_object(dataset)

    def create_subset_database(
        self,
        output_path: str,
        targets: List[str] = None,
        target_zaids: List[int] = None,
        quantity: str = "DA",
        projectile: str = "n",
        mt: int = None,
    ) -> int:
        """
        Create a new SQLite database with a subset of the data.

        Parameters
        ----------
        output_path : str
            Path for the new database file
        targets : List[str], optional
            List of targets in database format (e.g., ["Fe-0", "Fe-56"])
        target_zaids : List[int], optional
            List of target ZAIDs (e.g., [26000, 26056])
        quantity : str, optional
            Quantity type filter (default: "DA")
        projectile : str, optional
            Projectile filter (default: "n")
        mt : int, optional
            ENDF MT number filter (e.g., 2 for elastic)

        Returns
        -------
        int
            Number of datasets copied

        Examples
        --------
        >>> db = X4ProDatabase()
        >>> count = db.create_subset_database(
        ...     "iron_angular.db",
        ...     target_zaids=[26000, 26056],
        ...     mt=2
        ... )
        >>> print(f"Created database with {count} datasets")
        """
        import os

        # Convert ZAIDs to target patterns if provided
        target_patterns = []
        if targets:
            target_patterns.extend([self._normalize_target(t) for t in targets])
        if target_zaids:
            for zaid in target_zaids:
                pattern = _zaid_to_target_pattern(zaid)
                if pattern and pattern not in target_patterns:
                    target_patterns.append(pattern)

        if not target_patterns:
            raise ValueError("Must provide either targets or target_zaids")

        # Get source connection
        source_conn = self._get_connection()

        # Remove existing output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)

        # Create new database with same schema
        dest_conn = sqlite3.connect(output_path)
        dest_conn.row_factory = sqlite3.Row

        # Create tables with same schema
        dest_conn.execute("""
            CREATE TABLE IF NOT EXISTS x4pro_ds (
                DatasetID TEXT PRIMARY KEY,
                year1 INTEGER,
                author1 TEXT,
                Targ1 TEXT,
                Proj TEXT,
                MF INTEGER,
                MT INTEGER,
                ndat INTEGER,
                quant1 TEXT,
                reacode TEXT
            )
        """)
        dest_conn.execute("""
            CREATE TABLE IF NOT EXISTS x4pro_x5z (
                DatasetID TEXT PRIMARY KEY,
                jx5z TEXT
            )
        """)

        # Build query conditions for each target
        copied_count = 0
        for target_pattern in target_patterns:
            conditions = ["Targ1 = ?"]
            params = [target_pattern]

            # Add quantity filter
            if quantity:
                conditions.append("quant1 LIKE ?")
                params.append(f"%{quantity}%")

            # Add projectile filter
            proj_lower = projectile.lower()
            conditions.append("(Proj = ? OR Proj = ?)")
            params.extend([proj_lower, projectile.upper()])

            # Add MT filter
            if mt is not None:
                conditions.append("MT = ?")
                params.append(mt)

            # Query matching dataset IDs
            query = f"""
                SELECT DatasetID FROM x4pro_ds
                WHERE {' AND '.join(conditions)}
            """
            cursor = source_conn.execute(query, params)
            dataset_ids = [row[0] for row in cursor.fetchall()]

            # Copy data for each dataset
            for ds_id in dataset_ids:
                # Copy metadata row
                cursor = source_conn.execute(
                    "SELECT * FROM x4pro_ds WHERE DatasetID = ?", (ds_id,)
                )
                row = cursor.fetchone()
                if row:
                    dest_conn.execute(
                        """INSERT OR REPLACE INTO x4pro_ds
                           (DatasetID, year1, author1, Targ1, Proj, MF, MT, ndat, quant1, reacode)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (row["DatasetID"], row["year1"], row["author1"], row["Targ1"],
                         row["Proj"], row["MF"], row["MT"], row["ndat"], row["quant1"],
                         row["reacode"])
                    )

                # Copy JSON data row
                cursor = source_conn.execute(
                    "SELECT * FROM x4pro_x5z WHERE DatasetID = ?", (ds_id,)
                )
                row = cursor.fetchone()
                if row:
                    dest_conn.execute(
                        "INSERT OR REPLACE INTO x4pro_x5z (DatasetID, jx5z) VALUES (?, ?)",
                        (row["DatasetID"], row["jx5z"])
                    )

                copied_count += 1

        dest_conn.commit()
        dest_conn.close()

        return copied_count

    def _normalize_target(self, target: Union[str, int]) -> str:
        """
        Normalize target input to database format (e.g., "Fe-56").

        Accepts: "Fe56", "Fe-56", 26056, "26-FE-56", "Fe" (natural element)
        Returns: "Fe-56" or "Fe-0" for natural elements (database format)
        """
        if isinstance(target, int):
            # ZAID format
            return _zaid_to_target_pattern(target)

        target_str = str(target)

        # Already in database format "Fe-56" or "Fe-0"
        if re.match(r"^[A-Za-z]+-\d+$", target_str):
            return target_str.capitalize().replace(target_str[0], target_str[0].upper(), 1)

        # "Fe56" format -> "Fe-56"
        match = re.match(r"^([A-Za-z]+)(\d+)$", target_str)
        if match:
            elem = match.group(1).capitalize()
            mass = match.group(2)
            return f"{elem}-{mass}"

        # EXFOR format "26-FE-56"
        match = re.match(r"^(\d+)-([A-Za-z]+)-(\d+)$", target_str)
        if match:
            elem = match.group(2).capitalize()
            mass = match.group(3)
            return f"{elem}-{mass}"

        # Bare element symbol "Fe" -> "Fe-0" (natural element)
        # Validate it's a known element symbol
        if re.match(r"^[A-Za-z]{1,2}$", target_str):
            elem = target_str.capitalize()
            if elem in SYMBOL_TO_ATOMIC_NUMBER:
                return f"{elem}-0"

        return target_str


def read_exfor_from_database(
    db_path: str = None,
    target: str = None,
    target_zaid: int = None,
    projectile: str = "N",
    mt: int = None,
    energy_range: Tuple[float, float] = None,
) -> List["ExforAngularDistribution"]:
    """
    Convenience function to read EXFOR data from the X4Pro database.

    Parameters
    ----------
    db_path : str, optional
        Path to database. Uses KIKA_X4PRO_DB_PATH env var or default if None.
    target : str, optional
        Target in EXFOR notation (e.g., "26-FE-56")
    target_zaid : int, optional
        Target ZAID (e.g., 26056)
    projectile : str, optional
        Projectile (default: "N")
    mt : int, optional
        ENDF MT number
    energy_range : Tuple[float, float], optional
        Energy range (min, max) in MeV

    Returns
    -------
    List[ExforAngularDistribution]
        List of angular distribution datasets

    Examples
    --------
    >>> from kika.exfor.database import read_exfor_from_database
    >>> datasets = read_exfor_from_database(target_zaid=26056, mt=2)
    >>> for ds in datasets:
    ...     print(f"{ds.label}: {len(ds.energies())} energies")
    """
    with X4ProDatabase(db_path) as db:
        return db.query_angular_distributions(
            target=target,
            target_zaid=target_zaid,
            projectile=projectile,
            mt=mt,
            energy_range=energy_range,
        )
