import numpy as np


def control_inputs(x, t):
    """Return the four rotor RPM commands for the open-loop test case."""
    trim = 3290.0

    pitch_cmd = 0.0
    roll_cmd = 0.0
    climb_cmd = 0.0
    yaw_cmd = 0.0

    if t < 11.0:
        climb_cmd = 500.0

    if t > 8.0:
        yaw_cmd = -100.0
    if t > 9.0:
        yaw_cmd = 100.0
    if t > 10.0:
        yaw_cmd = 0.0

    if t > 12.0:
        yaw_cmd = 150.0
    if t > 13.0:
        yaw_cmd = -150.0
    if t > 14.0:
        yaw_cmd = 0.0

    if t > 16.0:
        climb_cmd = 150.0

    u = np.zeros(4)
    u[0] = trim + (pitch_cmd + roll_cmd + climb_cmd - yaw_cmd) / 4.0
    u[1] = trim + (-pitch_cmd - roll_cmd + climb_cmd - yaw_cmd) / 4.0
    u[2] = trim + (pitch_cmd - roll_cmd + climb_cmd + yaw_cmd) / 4.0
    u[3] = trim + (-pitch_cmd + roll_cmd + climb_cmd + yaw_cmd) / 4.0

    return u
