#!/usr/bin/env python3
"""
EXFOR JSON Schema Migration Script

Migrates EXFOR JSON files from the legacy format to the standardized v1.0 schema.

Key changes:
- Adds schema_version: "1.0.0"
- Renames 'result' -> 'cross_section' in data points
- Renames 'error_stat' -> 'uncertainty_stat'
- Renames 'energy_resolution_inputs' -> 'energy_resolution_input'
- Adds 'target_zaid' computed from reaction string
- Converts reaction string to structured dict
- Normalizes unit names (e.g., 'ADEG' -> 'deg', 'B/SR' -> 'b/sr')
- Renames 'E' -> 'value' in energy blocks
- Keeps original units (no value conversion)

Usage:
    python migrate_exfor_json.py --input /path/to/data/ --output /path/to/data_v1/
    python migrate_exfor_json.py --input /path/to/single_file.json --output /path/to/output.json

Author: KIKA Development Team
"""

import argparse
import json
import os
import re
import glob
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


SCHEMA_VERSION = "1.0.0"


def parse_reaction_notation(reaction: str) -> Dict[str, Any]:
    """
    Parse EXFOR reaction notation to extract structured information.

    Examples:
        "26-FE-56(N,EL)26-FE-56" -> {target: "Fe56", target_zaid: 26056, ...}
        "26-FE-0(N,EL)26-FE-0" -> {target: "Fe0", target_zaid: 26000, ...}

    Parameters:
        reaction: EXFOR reaction string

    Returns:
        Dictionary with parsed reaction information
    """
    result = {
        "target": "Unknown",
        "target_zaid": 0,
        "projectile": "n",
        "process": "EL",
        "notation": reaction,
    }

    # Parse target: ZZ-SY-AAA
    target_match = re.match(r"(\d+)-([A-Z]+)-(\d+)", reaction)
    if target_match:
        z = int(target_match.group(1))
        symbol = target_match.group(2).capitalize()
        a = int(target_match.group(3))

        result["target"] = f"{symbol}{a}" if a > 0 else f"{symbol}0"
        result["target_zaid"] = z * 1000 + a

    # Parse projectile
    proj_match = re.search(r"\(([A-Z]+),", reaction)
    if proj_match:
        result["projectile"] = proj_match.group(1).lower()

    # Parse process
    proc_match = re.search(r",([A-Z]+)\)", reaction)
    if proc_match:
        result["process"] = proc_match.group(1)

    return result


def normalize_unit(unit: str, unit_type: str) -> str:
    """
    Normalize unit string to standard format.

    Parameters:
        unit: Original unit string
        unit_type: Type of unit ('energy', 'angle', 'cross_section')

    Returns:
        Normalized unit string
    """
    unit_upper = unit.upper()

    if unit_type == "energy":
        mapping = {
            "EV": "eV",
            "KEV": "keV",
            "MEV": "MeV",
            "M": "MeV",  # Some files use 'M' for MeV
        }
        return mapping.get(unit_upper, unit)

    elif unit_type == "angle":
        mapping = {
            "ADEG": "deg",
            "DEG": "deg",
            "COS": "cos",
        }
        return mapping.get(unit_upper, unit)

    elif unit_type == "cross_section":
        mapping = {
            "B/SR": "b/sr",
            "MB/SR": "mb/sr",
            "UB/SR": "ub/sr",
            "MUB/SR": "ub/sr",
        }
        return mapping.get(unit_upper, unit)

    return unit


def migrate_data_point(point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a single data point to v1.0 schema.

    Changes:
        - result -> cross_section
        - error_stat -> uncertainty_stat
        - error_sys -> uncertainty_sys (if present)
    """
    migrated = {
        "angle": point.get("angle", 0.0),
        "cross_section": point.get("cross_section") or point.get("result", 0.0),
        "uncertainty_stat": point.get("uncertainty_stat") or point.get("error_stat", 0.0),
    }

    # Handle systematic uncertainty
    uncertainty_sys = point.get("uncertainty_sys") or point.get("error_sys")
    if uncertainty_sys is not None:
        migrated["uncertainty_sys"] = uncertainty_sys

    # Handle series
    if point.get("series"):
        migrated["series"] = point["series"]

    return migrated


def migrate_energy_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate an energy block to v1.0 schema.

    Changes:
        - E -> value
        - data points migrated
    """
    migrated = {
        "value": block.get("value") or block.get("E", 0.0),
        "data": [migrate_data_point(p) for p in block.get("data", [])],
    }

    if block.get("uncertainty"):
        migrated["uncertainty"] = block["uncertainty"]

    return migrated


def migrate_units(units: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate units to v1.0 schema with normalized names.
    """
    return {
        "energy": normalize_unit(units.get("energy", "MeV"), "energy"),
        "angle": normalize_unit(units.get("angle", "deg"), "angle"),
        "cross_section": normalize_unit(
            units.get("cross_section") or units.get("dsig", "b/sr"),
            "cross_section"
        ),
    }


def migrate_method(method: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate method section to v1.0 schema.

    Changes:
        - energy_resolution_inputs -> energy_resolution_input
        - Simplifies flight_path info into notes
    """
    migrated = {
        "type": method.get("type", "Unknown"),
    }

    # Handle energy resolution input (singular vs plural)
    eri = method.get("energy_resolution_input") or method.get("energy_resolution_inputs")
    if eri:
        migrated["energy_resolution_input"] = eri

    # Preserve notes
    if method.get("notes"):
        migrated["notes"] = method["notes"]

    # If there's flight_path info but no notes, create a note
    if method.get("flight_path") and not migrated.get("notes"):
        fp = method["flight_path"]
        note_parts = []
        for key, value in fp.items():
            if isinstance(value, dict) and "value" in value:
                note_parts.append(f"{key}: {value['value']} {value.get('unit', '')}")
        if note_parts:
            migrated["notes"] = "Flight path: " + ", ".join(note_parts)

    return migrated


def migrate_json_file(input_path: str, output_path: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Migrate a single EXFOR JSON file to v1.0 schema.

    Parameters:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        verbose: Print progress messages

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        with open(input_path, "r") as f:
            data = json.load(f)

        # Check if already migrated
        if data.get("schema_version") == SCHEMA_VERSION:
            return True, f"Already at v{SCHEMA_VERSION}"

        # Start building migrated structure
        migrated = {
            "schema_version": SCHEMA_VERSION,
            "entry": data.get("entry", "Unknown"),
            "subentry": data.get("subentry", "Unknown"),
        }

        # Migrate citation
        if "citation" in data:
            migrated["citation"] = data["citation"]

        # Migrate reaction (convert string to structured dict)
        reaction = data.get("reaction")
        if isinstance(reaction, str):
            migrated["reaction"] = parse_reaction_notation(reaction)
        elif isinstance(reaction, dict):
            # Already structured, just copy
            migrated["reaction"] = reaction
        else:
            migrated["reaction"] = {
                "target": "Unknown",
                "target_zaid": 0,
                "projectile": "n",
                "process": "EL",
                "notation": str(reaction) if reaction else "",
            }

        # Copy facility
        if "facility" in data:
            migrated["facility"] = data["facility"]

        # Migrate method
        if "method" in data:
            migrated["method"] = migrate_method(data["method"])

        # Copy detector and sample
        if "detector" in data:
            migrated["detector"] = data["detector"]
        if "sample" in data:
            migrated["sample"] = data["sample"]

        # Copy quantity
        migrated["quantity"] = data.get("quantity", "DA")

        # Copy angle frame
        migrated["angle_frame"] = data.get("angle_frame", "LAB").upper()

        # Migrate units
        if "units" in data:
            migrated["units"] = migrate_units(data["units"])
        else:
            migrated["units"] = {"energy": "MeV", "angle": "deg", "cross_section": "b/sr"}

        # Migrate energy blocks
        migrated["energies"] = [
            migrate_energy_block(block)
            for block in data.get("energies", [])
        ]

        # Write output
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(migrated, f, indent=2)

        n_energies = len(migrated["energies"])
        n_points = sum(len(e["data"]) for e in migrated["energies"])
        return True, f"Migrated {n_energies} energies, {n_points} data points"

    except Exception as e:
        return False, f"Error: {e}"


def migrate_directory(input_dir: str, output_dir: str, pattern: str = "*.json",
                      verbose: bool = False) -> Dict[str, Any]:
    """
    Migrate all EXFOR JSON files in a directory.

    Parameters:
        input_dir: Input directory path
        output_dir: Output directory path
        pattern: Glob pattern for files
        verbose: Print progress

    Returns:
        Dictionary with migration statistics
    """
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "files": [],
    }

    json_files = glob.glob(os.path.join(input_dir, pattern))
    stats["total"] = len(json_files)

    if verbose:
        print(f"Found {len(json_files)} files to migrate")

    for input_path in json_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        success, message = migrate_json_file(input_path, output_path, verbose)

        file_info = {
            "filename": filename,
            "success": success,
            "message": message,
        }
        stats["files"].append(file_info)

        if success:
            if "Already" in message:
                stats["skipped"] += 1
            else:
                stats["success"] += 1
        else:
            stats["failed"] += 1

        if verbose:
            status = "OK" if success else "FAILED"
            print(f"  [{status}] {filename}: {message}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate EXFOR JSON files to standardized v1.0 schema"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file or directory"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*.json",
        help="Glob pattern for files (default: *.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - no files will be written")

    if os.path.isfile(args.input):
        # Single file migration
        if args.dry_run:
            print(f"Would migrate: {args.input} -> {args.output}")
        else:
            success, message = migrate_json_file(args.input, args.output, args.verbose)
            status = "SUCCESS" if success else "FAILED"
            print(f"{status}: {message}")
    else:
        # Directory migration
        if args.dry_run:
            json_files = glob.glob(os.path.join(args.input, args.pattern))
            print(f"Would migrate {len(json_files)} files from {args.input} to {args.output}")
            for f in json_files:
                print(f"  {os.path.basename(f)}")
        else:
            stats = migrate_directory(args.input, args.output, args.pattern, args.verbose)

            print(f"\nMigration complete:")
            print(f"  Total files: {stats['total']}")
            print(f"  Migrated:    {stats['success']}")
            print(f"  Skipped:     {stats['skipped']}")
            print(f"  Failed:      {stats['failed']}")

            if stats["failed"] > 0:
                print("\nFailed files:")
                for f in stats["files"]:
                    if not f["success"]:
                        print(f"  {f['filename']}: {f['message']}")


if __name__ == "__main__":
    main()
