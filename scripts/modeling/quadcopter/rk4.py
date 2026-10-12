import numpy as np

from state_derivative import state_derivative


def rk4(x, u, dt):
    """Advance the state one RK4 step."""
    k1 = state_derivative(x, u)
    k2 = state_derivative(x + 0.5 * dt * k1, u)
    k3 = state_derivative(x + 0.5 * dt * k2, u)
    k4 = state_derivative(x + dt * k3, u)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next
