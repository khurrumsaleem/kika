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
