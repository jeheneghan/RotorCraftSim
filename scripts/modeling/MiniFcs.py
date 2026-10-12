"""Small PID-based body-rate controller for the quadrotor simulator."""

import numpy as np


class _RatePid:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_measurement = None

    def reset(self):
        self.integral = 0.0
        self.previous_measurement = None

    def update(self, setpoint, measurement, dt):
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")

        error = setpoint - measurement
        if self.previous_measurement is None:
            measurement_rate = 0.0
        else:
            measurement_rate = (
                measurement - self.previous_measurement
            ) / dt
        self.previous_measurement = measurement

        output = (
            self.kp * error
            + self.ki * self.integral
            - self.kd * measurement_rate
        )
        limited_output = float(
            np.clip(output, -self.output_limit, self.output_limit)
        )

        if output == limited_output or error * output <= 0.0:
            self.integral += error * dt

        return limited_output


class MiniFcs:
    """Convert roll, pitch, and yaw rate targets into motor RPM offsets.

    Rate inputs and measurements are body rates ``[p, q, r]`` in radians per
    second. The returned ``[roll, pitch, yaw]`` corrections are in RPM.
    """

    def __init__(self):
        self._controllers = (
            _RatePid(kp=100.0, ki=50.0, kd=10.0, output_limit=1800.0),
            _RatePid(kp=100.0, ki=5.0, kd=10.0, output_limit=1800.0),
            _RatePid(kp=1600.0, ki=100.0, kd=100.0, output_limit=2500.0),
        )

    def reset(self):
        """Clear the rate-controller integrators and derivative history."""
        for controller in self._controllers:
            controller.reset()

    def update(self, rate_setpoint, body_rates, dt):
        """Return roll, pitch, and yaw motor corrections in RPM."""
        setpoints = np.asarray(rate_setpoint, dtype=float)
        measurements = np.asarray(body_rates, dtype=float)
        if setpoints.shape != (3,) or measurements.shape != (3,):
            raise ValueError("rate_setpoint and body_rates must each have shape (3,)")

        return np.array(
            [
                controller.update(setpoint, measurement, dt)
                for controller, setpoint, measurement in zip(
                    self._controllers, setpoints, measurements
                )
            ],
            dtype=float,
        )
