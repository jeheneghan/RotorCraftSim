from RotorParams import RotorParams
import copy

class AscTecPelicanParams:
    def __init__(self):
        self.weight = 2.8 # lbf
        self.inertia = [0.032, 0.032, 0.052]  # slug-ft^2
        self.rotor_params_FR = RotorParams()
        self.rotor_params_FR.radius = 0.42  # ft
        self.rotor_params_FR.lift_curve_slope = 5.7 # per rad
        self.rotor_params_FR.num_blades = 2
        self.rotor_params_FR.blade_chord = 0.09  # ft
        self.rotor_params_FR.rotor_inertia = 3.0e-6 # slug-ft^2
        self.rotor_params_FR.zero_lift_drag_coeff = 0.01
        self.rotor_params_FR.root_blade_pitch_range = [0.49, 0.49] # rad
        self.rotor_params_FR.blade_twist = -0.33 # rad
        self.rotor_params_FR.thrust_direction = [0, 0, -1]  # unit vector in body frame
        self.rotor_params_FR.rotor_location = [0.49, 0.49, 0] # ft, location of rotor in body frame
        self.rotor_params_FR.rotor_spin_direction = 1  # 1 for clockwise, -1 for counterclockwise
        self.rotor_params_BR = copy.deepcopy(self.rotor_params_FR)
        self.rotor_params_BR.rotor_location = [-0.49, 0.49, 0]
        self.rotor_params_BR.rotor_spin_direction = -1
        self.rotor_params_BL = copy.deepcopy(self.rotor_params_FR)
        self.rotor_params_BL.rotor_location = [-0.49, -0.49, 0]
        self.rotor_params_BL.rotor_spin_direction = 1
        self.rotor_params_FL = copy.deepcopy(self.rotor_params_FR)
        self.rotor_params_FL.rotor_location = [0.49, -0.49, 0]
        self.rotor_params_FL.rotor_spin_direction = -1