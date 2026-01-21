"""
EXFOR-specific constants for the kika library.
"""

# Valid angle frames
FRAME_LAB = "LAB"
FRAME_CM = "CM"
VALID_FRAMES = {FRAME_LAB, FRAME_CM}

# Valid angle units
ANGLE_UNIT_DEG = "deg"
ANGLE_UNIT_COS = "cos"
VALID_ANGLE_UNITS = {ANGLE_UNIT_DEG, ANGLE_UNIT_COS}

# Valid energy units
ENERGY_UNIT_EV = "eV"
ENERGY_UNIT_KEV = "keV"
ENERGY_UNIT_MEV = "MeV"
VALID_ENERGY_UNITS = {ENERGY_UNIT_EV, ENERGY_UNIT_KEV, ENERGY_UNIT_MEV}

# Valid cross section units
XS_UNIT_B_SR = "b/sr"
XS_UNIT_MB_SR = "mb/sr"
XS_UNIT_UB_SR = "ub/sr"
VALID_XS_UNITS = {XS_UNIT_B_SR, XS_UNIT_MB_SR, XS_UNIT_UB_SR}

# Unit conversion factors (to base units: MeV, b/sr, deg)
ENERGY_TO_MEV = {
    "eV": 1e-6,
    "keV": 1e-3,
    "MeV": 1.0,
}

XS_TO_B_SR = {
    "b/sr": 1.0,
    "mb/sr": 1e-3,
    "ub/sr": 1e-6,
}

# Schema version for standardized JSON format
SCHEMA_VERSION = "1.0.0"

# Neutron mass in atomic mass units
NEUTRON_MASS_AMU = 1.008665

# Energy matching tolerance (MeV) for grouping data by energy
ENERGY_MATCH_ABS_TOL = 1e-6

# Default absolute energy tolerance for plotting (MeV)
# Used in to_plot_data() for matching experimental data to requested energy
PLOTTING_ENERGY_TOLERANCE = 0.01  # 10 keV

# =============================================================================
# X4Pro Database Configuration
# =============================================================================

# Environment variable for database path
DB_PATH_ENV_VAR = "KIKA_X4PRO_DB_PATH"

# Default database path (None = must be set via env var or passed explicitly)
# Users should set KIKA_X4PRO_DB_PATH environment variable or pass db_path to functions
DB_DEFAULT_PATH = None

# Database unit conversion factors
# Maps database unit strings to conversion factors (to base unit)
DB_UNIT_MAPPINGS = {
    "energy": {
        # Energy units -> eV conversion factor
        "EV": 1.0,
        "KEV": 1e3,
        "MEV": 1e6,
        "GEV": 1e9,
    },
    "cross_section": {
        # Cross section units -> b/sr conversion factor
        "B/SR": 1.0,
        "MB/SR": 1e-3,
        "MUB/SR": 1e-6,  # microbarns
        "UB/SR": 1e-6,
    },
}

# Database family code mappings
DB_FAMILY_MAPPINGS = {
    "energy": ["EN", "E", "EN-CM"],
    "angle": ["ANG", "ANG-CM"],
    "cosine": ["COS", "COS-CM"],
    "cross_section": ["Data", "DATA"],
    "uncertainty": ["dData", "DATA-ERR", "ERR-T", "ERR-S"],
}

# Database quantity codes for angular distributions
DB_DA_QUANTITIES = ["DA", "DA,,RTH", "DA/DA", "DA/DE"]
