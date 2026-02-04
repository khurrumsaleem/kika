"""
Backward compatibility module - transforms are now in ExforAngularDistribution.

LAB <-> CM frame conversion utilities for angular distributions.

The transformations assume non-relativistic kinematics, valid for neutron
scattering on heavy nuclei at energies below ~20 MeV.

Key relations:
    alpha = m_projectile / m_target

    LAB to CM angle: mu_CM = -alpha*(1 - mu_L^2) + mu_L*sqrt(1 - alpha^2*(1 - mu_L^2))

    Jacobian (dOmega_CM/dOmega_LAB) = (1 + alpha^2 + 2*alpha*mu_CM)^(3/2) / |1 + alpha*mu_CM|

    Cross section: (dsigma/dOmega)_CM = (dsigma/dOmega)_LAB / Jacobian

Note:
    All transform functions are now available as static methods on
    ExforAngularDistribution. This module re-exports them for backward
    compatibility.

    Example:
        # Old way (still works)
        from kika.exfor.transforms import transform_lab_to_cm

        # New way
        from kika.exfor import ExforAngularDistribution
        ExforAngularDistribution.transform_lab_to_cm(...)
"""

from kika.exfor.angular_distribution import ExforAngularDistribution

# Re-export static methods for backward compatibility
cos_cm_from_cos_lab = ExforAngularDistribution.cos_cm_from_cos_lab
cos_lab_from_cos_cm = ExforAngularDistribution.cos_lab_from_cos_cm
jacobian_cm_to_lab = ExforAngularDistribution.jacobian_cm_to_lab
jacobian_lab_to_cm = ExforAngularDistribution.jacobian_lab_to_cm
transform_lab_to_cm = ExforAngularDistribution.transform_lab_to_cm
transform_cm_to_lab = ExforAngularDistribution.transform_cm_to_lab
angle_deg_to_cos = ExforAngularDistribution.angle_deg_to_cos
cos_to_angle_deg = ExforAngularDistribution.cos_to_angle_deg
