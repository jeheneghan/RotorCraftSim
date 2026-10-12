class RotorParams:
    def __init__(self):
        self.radius = 5.1  # ft
        self.lift_curve_slope = 5.7 # per rad
        self.num_blades = 2
        self.blade_chord = 0.43  # ft
        self.hinge_offset = 0.2 # ft
        self.inertia_of_blade_about_flapping_hinge = 1.0 # slug-ft^2
        self.rotor_inertia = 3.0e-6 # slug-ft^2
        self.zero_lift_drag_coeff = 0.01
        self.root_blade_pitch_range = [0.12, 0.30] # rad
        self.blade_twist = -0.1 # rad
        self.thrust_direction = [0, 0, -1]  # unit vector in body frame
        self.rotor_location = [0.49, 0.49, 0] # ft, location of rotor in body frame
        self.rotor_spin_direction = 1  # 1 for clockwise, -1 for counterclockwise