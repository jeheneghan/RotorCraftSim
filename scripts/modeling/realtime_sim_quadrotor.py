"""
Real-time quadrotor simulation using the quadrotor state derivative model.

This keeps the same threaded physics/graphics architecture as the existing
realtime simulator, but swaps the equations of motion to the quadrotor
state_derivative implementation and maps the controls to an Xbox-style
joystick layout.
"""

import sys
import time
import threading
from pathlib import Path

import keyboard
import numpy as np
import pygame
from scipy.optimize import brentq

from pfd.primary_flight_display import AircraftState, PrimaryFlightDisplay

QUADROTOR_DIR = Path(__file__).resolve().parent / "quadcopter"
if str(QUADROTOR_DIR) not in sys.path:
    sys.path.insert(0, str(QUADROTOR_DIR))

from modeling.quadcopter.state_derivative import state_derivative
from modeling.quadcopter.AscTecPelicanParams import AscTecPelicanParams
from modeling.quadcopter.fthrust_unity import fthrust_unity
from modeling.MiniFcs import MiniFcs


PHYSICS_HZ = 200
GRAPHICS_FPS = 60
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800

PHYSICS_DT = 1.0 / PHYSICS_HZ
GRAPHICS_DT = 1.0 / GRAPHICS_FPS

SIM_STATE = {
    "running": True,
}

AIRCRAFT_PARAMS = AscTecPelicanParams()
ROTOR_PARAMS = AIRCRAFT_PARAMS.rotor_params_FR
RPM_TO_RADIANS_PER_SECOND = 2.0 * np.pi / 60.0
MAX_MOTOR_RPM = 22000.0
MAX_CLIMB_RATE_FPS = 40.0
MAX_DESCENT_RATE_FPS = 20.0
CLIMB_RATE_FEEDBACK_RPM_PER_FPS = 60.0
MAX_ROLL_RATE_DPS = 90.0
MAX_PITCH_RATE_DPS = 90.0
MAX_YAW_RATE_DPS = 45.0


def _find_rpm_for_climb_rate(climb_rate_fps):
    """Find the rotor speed that balances weight at a steady vertical rate."""
    target_thrust = AIRCRAFT_PARAMS.weight / 4.0
    theta_0 = ROTOR_PARAMS.root_blade_pitch_range[0]

    return brentq(
        lambda rpm: fthrust_unity(
            0.0,
            np.array([0.0, 0.0, -climb_rate_fps]),
            rpm * RPM_TO_RADIANS_PER_SECOND,
            theta_0,
            ROTOR_PARAMS,
        )[0]
        - target_thrust,
        1.0,
        30000.0,
    )


HOVER_RPM = _find_rpm_for_climb_rate(0.0)
MAX_CLIMB_RPM = _find_rpm_for_climb_rate(MAX_CLIMB_RATE_FPS)
MAX_DESCENT_RPM = _find_rpm_for_climb_rate(-MAX_DESCENT_RATE_FPS)


class XboxQuadrotorControls:
    """Container for the quadrotor control input vector."""

    def __init__(self):
        self.u = np.full(4, HOVER_RPM, dtype=float)


def _state_vector_to_aircraft_state(x_state, controls_state):
    """Convert the quadrotor state vector to the PFD display state."""
    roll_rad = x_state[6]
    pitch_rad = x_state[7]
    heading_rad = x_state[8]
    altitude_m = x_state[11]
    airspeed = float(np.linalg.norm(x_state[0:3]))

    return AircraftState(
        pitch=np.rad2deg(pitch_rad),
        roll=np.rad2deg(roll_rad),
        airspeed=airspeed * 1.943844,  # m/s -> knots
        airspeed_cmd=0.0,
        vspeed=0.0,
        altitude=altitude_m * 3.28084,  # m -> ft
        altitude_cmd=0.0,
        heading=np.rad2deg(heading_rad),
        heading_cmd=0.0,
        course=np.rad2deg(heading_rad),
        power=float(np.mean(controls_state.u) / MAX_MOTOR_RPM * 100.0),
    )


def _rk4_step(x, u, dt):
    """Advance the quadrotor state one RK4 step using state_derivative."""
    k1 = state_derivative(x, u, AIRCRAFT_PARAMS)
    k2 = state_derivative(x + 0.5 * dt * k1, u, AIRCRAFT_PARAMS)
    k3 = state_derivative(x + 0.5 * dt * k2, u, AIRCRAFT_PARAMS)
    k4 = state_derivative(x + dt * k3, u, AIRCRAFT_PARAMS)

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _update_controls_from_xbox(joystick, controls_state, x_state, fcs, dt):
    """Map Xbox controller axes into quadrotor rotor RPM commands.

    Controller axes 2/3 command roll/pitch rates; axes 0/1 command yaw and
    climb rates, respectively.
    """
    rate_setpoint = np.deg2rad(
        [
            joystick.get_axis(2) * MAX_ROLL_RATE_DPS,
            joystick.get_axis(3) * MAX_PITCH_RATE_DPS,
            joystick.get_axis(0) * MAX_YAW_RATE_DPS,
        ]
    )
    roll_cmd, pitch_cmd, yaw_cmd = fcs.update(
        rate_setpoint, x_state[3:6], dt
    )

    collective_delta = joystick.get_axis(1)
    if collective_delta >= 0.0:
        climb_rate_cmd_fps = collective_delta * MAX_CLIMB_RATE_FPS
    else:
        climb_rate_cmd_fps = collective_delta * MAX_DESCENT_RATE_FPS
    if climb_rate_cmd_fps >= 0.0:
        collective_trim = HOVER_RPM + (
            MAX_CLIMB_RPM - HOVER_RPM
        ) * climb_rate_cmd_fps / MAX_CLIMB_RATE_FPS
    else:
        collective_trim = MAX_DESCENT_RPM + (
            HOVER_RPM - MAX_DESCENT_RPM
        ) * (climb_rate_cmd_fps + MAX_DESCENT_RATE_FPS) / MAX_DESCENT_RATE_FPS

    climb_rate = (
        np.sin(x_state[7]) * x_state[0]
        - np.sin(x_state[6]) * np.cos(x_state[7]) * x_state[1]
        - np.cos(x_state[6]) * np.cos(x_state[7]) * x_state[2]
    ) * 3.28084
    collective_trim += CLIMB_RATE_FEEDBACK_RPM_PER_FPS * (
        climb_rate_cmd_fps - climb_rate
    )

    # Motor order is FR, BR, BL, FL; signs match their locations and spin directions.
    u = np.array(
        [
            collective_trim + pitch_cmd - roll_cmd + yaw_cmd,
            collective_trim - pitch_cmd - roll_cmd - yaw_cmd,
            collective_trim - pitch_cmd + roll_cmd + yaw_cmd,
            collective_trim + pitch_cmd + roll_cmd - yaw_cmd,
        ],
        dtype=float,
    )

    controls_state.u[:] = np.clip(u, 0.0, MAX_MOTOR_RPM)


def _check_exit_condition():
    if keyboard.is_pressed("e") or keyboard.is_pressed("esc"):
        print("Exit key pressed. Stopping simulation.")
        SIM_STATE["running"] = False


def simulate_realtime_quadrotor(X0, controls_state):
    """Run a real-time quadrotor simulation with a PFD view."""
    shared_state = {
        "y": np.array(X0, dtype=float).reshape(-1, 1),
        "y_prev": np.array(X0, dtype=float).reshape(-1, 1).copy(),
        "aircraft_state": None,
        "running": True,
    }

    pfd = PrimaryFlightDisplay(
        (SCREEN_WIDTH, SCREEN_HEIGHT),
        masked=True,
        max_fps=GRAPHICS_FPS,
    )

    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() == 0:
        print("No Xbox controller detected. Keyboard fallback is unavailable for this quadrotor simulator.")
        print("  Connect an Xbox controller and press 'e' or 'esc' to exit.")
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Using controller: {joystick.get_name()}")
        print("Xbox mapping:")
        print("  Axes 2/3 -> roll/pitch rate")
        print("  Axis 0 -> yaw rate")
        print("  Axis 1 -> climb/descent rate")
        print("  'e' or 'esc': Exit simulation")

    fcs = MiniFcs()
    def physics_loop():
        while SIM_STATE["running"]:
            loop_start = time.time()

            y_next = _rk4_step(shared_state["y"][:, 0], controls_state.u, PHYSICS_DT)
            shared_state["y_prev"] = shared_state["y"].copy()
            shared_state["y"] = y_next.reshape(-1, 1)
            shared_state["aircraft_state"] = _state_vector_to_aircraft_state(
                shared_state["y"][:, 0], controls_state
            )

            _check_exit_condition()
            if not SIM_STATE["running"]:
                break

            elapsed = time.time() - loop_start
            sleep_time = PHYSICS_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    physics_thread = threading.Thread(target=physics_loop, daemon=True)
    physics_thread.start()

    start_time = time.time()
    previous_control_time = time.perf_counter()

    try:
        while shared_state["running"]:
            loop_start = time.time()
            elapsed = loop_start - start_time

            pygame.event.pump()
            if joystick is not None:
                control_time = time.perf_counter()
                control_dt = control_time - previous_control_time
                previous_control_time = control_time
                _update_controls_from_xbox(
                    joystick,
                    controls_state,
                    shared_state["y"][:, 0].copy(),
                    fcs,
                    control_dt,
                )

            if shared_state["aircraft_state"] is not None:
                pfd.update(shared_state["aircraft_state"], elapsed)
                pfd.draw()
                pfd.render()

            loop_elapsed = time.time() - loop_start
            sleep_time = GRAPHICS_DT - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if not physics_thread.is_alive():
                break
    finally:
        shared_state["running"] = False
        SIM_STATE["running"] = False
        physics_thread.join(timeout=1.0)
        print("Simulation complete.")


def main():
    x0 = np.zeros(12, dtype=float)
    x0[11] = 0.0
    controls_state = XboxQuadrotorControls()
    simulate_realtime_quadrotor(X0=x0, controls_state=controls_state)


if __name__ == "__main__":
    main()
