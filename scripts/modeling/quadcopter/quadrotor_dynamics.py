import numpy as np
import matplotlib.pyplot as plt

from control_inputs import control_inputs
from rk4 import rk4


def simulate_quadrotor(dt=0.02, simulation_time=30.0):
    """Run the open-loop quadrotor simulation used in QuadrotorDynamics.m."""
    t = np.arange(0.0, simulation_time + dt, dt)
    n_states = 12
    n_inputs = 4

    x = np.zeros((n_states, len(t) + 1))
    x[11, 0] = 0.0
    u = np.zeros((n_inputs, len(t)))

    for k in range(len(t)):
        u[:, k] = control_inputs(x[:, k], t[k])
        x[:, k + 1] = rk4(x[:, k], u[:, k], dt)

    return t, x, u


def plot_trajectory(t, x):
    plt.figure()
    plt.plot(np.degrees(x[8, :]), x[11, :],'-o')
    plt.title('Path of Drone (Open Loop Inputs)')
    plt.xlabel('Yaw Angle')
    plt.ylabel('Vertical Position')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    t, x, u = simulate_quadrotor()
    plot_trajectory(t, x)
