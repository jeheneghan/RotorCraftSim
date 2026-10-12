import numpy as np

from AscTecPelicanParams import AscTecPelicanParams
from fthrust_unity import fthrust_unity


DEFAULT_PARAMS = AscTecPelicanParams()
LBF_TO_NEWTONS = 4.4482216152605
SLUG_FT2_TO_KG_M2 = 1.3558179483314
FT_TO_METERS = 0.3048
METERS_PER_SECOND_TO_FEET_PER_SECOND = 1.0 / FT_TO_METERS
RPM_TO_RADIANS_PER_SECOND = 2.0 * np.pi / 60.0


def state_derivative(x, u, params=DEFAULT_PARAMS):
    """Return the 12-state quadrotor dynamics derivative.

    State vector x:
        [ub, vb, wb, p, q, r, phi, theta, psi, xE, yE, hE]
    Rotor inputs are RPM; state and derivative units are SI.
    """
    ub = x[0]
    vb = x[1]
    wb = x[2]
    p = x[3]
    q = x[4]
    r = x[5]
    phi = x[6]
    theta = x[7]
    psi = x[8]
    xE = x[9]
    yE = x[10]
    hE = x[11]

    m = params.weight * LBF_TO_NEWTONS / 9.81
    Ixx, Iyy, Izz = (
        inertia * SLUG_FT2_TO_KG_M2 for inertia in params.inertia
    )
    g = 9.81

    rotors = (
        params.rotor_params_FR,
        params.rotor_params_BR,
        params.rotor_params_BL,
        params.rotor_params_FL,
    )
    rotor_thrusts = []
    rotor_torques = []
    for rpm, rotor in zip(u, rotors):
        omega = rpm * RPM_TO_RADIANS_PER_SECOND * rotor.rotor_spin_direction
        if abs(omega) < 1e-6:
            rotor_thrusts.append(0.0)
            rotor_torques.append(0.0)
            continue

        thrust, torque, _ = fthrust_unity(
            0.0,
            x[0:3] * METERS_PER_SECOND_TO_FEET_PER_SECOND,
            omega,
            rotor.root_blade_pitch_range[0],
            rotor,
        )
        rotor_thrusts.append(thrust * LBF_TO_NEWTONS)
        rotor_torques.append(torque * LBF_TO_NEWTONS * FT_TO_METERS)

    F1, F2, F3, F4 = rotor_thrusts
    T1, T2, T3, T4 = rotor_torques

    Fz = F1 + F2 + F3 + F4
    L = sum(
        -thrust * rotor.rotor_location[1] * FT_TO_METERS
        for thrust, rotor in zip(rotor_thrusts, rotors)
    )
    M = sum(
        thrust * rotor.rotor_location[0] * FT_TO_METERS
        for thrust, rotor in zip(rotor_thrusts, rotors)
    )
    N = T1 + T2 + T3 + T4

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cthe = np.cos(theta)
    sthe = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    xdot = np.zeros(12)
    xdot[0] = -g * sthe + r * vb - q * wb
    xdot[1] = g * sphi * cthe - r * ub + p * wb
    xdot[2] = (1.0 / m) * (-Fz) + g * cphi * cthe + q * ub - p * vb
    xdot[3] = (1.0 / Ixx) * (L + (Iyy - Izz) * q * r)
    xdot[4] = (1.0 / Iyy) * (M + (Izz - Ixx) * p * r)
    xdot[5] = (1.0 / Izz) * (N + (Ixx - Iyy) * p * q)
    xdot[6] = p + (q * sphi + r * cphi) * sthe / cthe
    xdot[7] = q * cphi - r * sphi
    xdot[8] = (q * sphi + r * cphi) / cthe
    xdot[9] = cthe * cpsi * ub + (-cphi * spsi + sphi * sthe * cpsi) * vb + (sphi * spsi + cphi * sthe * cpsi) * wb
    xdot[10] = cthe * spsi * ub + (cphi * cpsi + sphi * sthe * spsi) * vb + (-sphi * cpsi + cphi * sthe * spsi) * wb
    xdot[11] = -1.0 * (-sthe * ub + sphi * cthe * vb + cphi * cthe * wb)

    return xdot
