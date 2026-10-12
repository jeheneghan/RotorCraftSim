import math
from scipy.optimize import fsolve

def fthrust_unity(vi0, V, w, theta_0, params):
    """Mirror the MATLAB FthrustUnity rotor model.

    Returns:
        thrust, torque
    """

    # Constants
    rho = 0.0023769  # slug/ft^3

    # Rotor parameters
    R = params.radius
    A = math.pi * R**2
    a = params.lift_curve_slope
    b = params.num_blades
    c = params.blade_chord
    theta_1 = params.blade_twist
    Cd0 = params.zero_lift_drag_coeff

    # Rotor state
    u_p = V[0]
    v_p = V[1]
    w_p = V[2]

    def thrust_equation(vi):
        return rho*a*b*c*R/4 * ((w_p - vi)*abs(w) + 2/3*(abs(w)*R)**2*(theta_0+3/4*theta_1)
                                     + (u_p**2 + v_p**2) * (theta_0/2 + 1/2*theta_1))

    def compute_thrust(vi):
        Vp = max(1e-6, math.sqrt(u_p**2 + v_p**2 + (w_p - vi)**2))
        T = thrust_equation(vi)
        return T - 2*rho*A*Vp*vi

    # Solve for thrust by finding the induced velocity vi that satisfies the thrust equation
    vi = vi0
    temp = fsolve(compute_thrust, vi)
    vi = temp[0]
    thrust = thrust_equation(vi)

    # Solve for torque
    pi = thrust*(vi - w_p)
    pp = rho*Cd0*b*c*abs(w)*R**2/8 * ((abs(w)*R)**2 + u_p**2 + v_p**2)
    torque = 1/w * (pi + pp)

    return thrust, torque, vi
