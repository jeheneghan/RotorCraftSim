from modeling.realtime_sim_quadrotor import (
    XboxQuadrotorControls,
    simulate_realtime_quadrotor,
)

import numpy as np

def _init_():
    X0 = np.zeros(12)
    U0 = XboxQuadrotorControls()
    return X0, U0

def main():

    X0, U0 = _init_()
    simulate_realtime_quadrotor(X0=X0, controls_state=U0)

if __name__ == "__main__":
    main()