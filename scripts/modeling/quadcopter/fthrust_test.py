import matplotlib.pyplot as plt
import numpy as np
from fthrust_unity import fthrust_unity
from AscTecPelicanParams import AscTecPelicanParams

params = AscTecPelicanParams()
rotor_params = params.rotor_params_FR

omega = np.linspace(10, 1000, 100)  # rad/s
V = np.array([50.0, 0.0, 5.0])

thrust = np.zeros_like(omega)
vi0 = 20 # fps

for i, w in enumerate(omega):
    theta_0 = rotor_params.root_blade_pitch_range[0]  # rad
    thrust[i], _, vi0 = fthrust_unity(vi0, V, w, theta_0, rotor_params)

plt.figure(figsize=(10, 6))
plt.plot(omega, thrust)
plt.xlabel('Rotor Speed (rad/s)')
plt.ylabel('Thrust')
plt.title('Thrust vs Rotor Speed')
plt.show()