import numpy as np
from scipy.optimize import root

g = 9.81

def _get_fr_coeffs(X):
    s2 = np.sqrt(2)
    A0 = 1/(1+X**2)
    B0 = 1 - A0
    A1 = (s2 * X**2 + X**(-s2)) / (X**s2 + X**(-s2))
    B1 = 1 - A1
    return A0, B0, A1, B1


def get_velocity(r, phi, v0, r0, nu, R):
    """Calculate the velocity profile, i.e. the flow velocity in tangential direction for an arbitrary r and phi.
    i.e. the solution of the navier-stokes equation"""
    s2 = np.sqrt(2)
    x = r / r0
    A0, B0, A1, B1 = _get_fr_coeffs(R/r0)
    return v0 * (A0*x + B0/x) - ( (A1 * x**s2) + (B1 * x**(-s2)) - x**2) * (g*r0**2)/(2*nu) * np.cos(phi)


def get_flow(phi, v0, r0, nu, R):
    s2 = np.sqrt(2)
    X = R/r0
    A0, B0, A1, B1 = _get_fr_coeffs(X)

    A = A0*(X**2-1)/2 + B0*np.log(X)
    B = A1 * (X**(1+s2)-1) / (1+s2) + B1 * (X**(1-s2)-1) / (1-s2) - (X**3-1)/3
    v_tilde = (g*r0**2)/(2*nu)
    return (v0 * A - B * v_tilde * np.cos(phi)) * r0


def _flow_root_func(x, phi, v0, r0, nu, R0):
    """Helper function to determine the maximum radius at each phi"""
    flow = v0 * r0 * ((R0 / r0) ** 2 - 1) / 2
    A = v0 * r0 * (x**2 - 1) / 2
    s2 = np.sqrt(2)
    B = (0.5 * x**(1+s2) + 0.5 * x**(1-s2) - (x**3)/3 - 2/3)
    C = (g*r0**3) / (2*nu) * np.cos(phi)
    return A + B*C - flow


def _get_R_outer(phi, v0, r0, nu, Rmax):
    """Calculate the outer perimeter of the honey-blob as a function of phi, using conservation of flow"""
    flow0 = get_flow(np.pi/2, v0, r0, nu, Rmax)
    res = root(lambda R: get_flow(phi, v0, r0, nu, R) - flow0, r0)
    if res.success:
        return res.x[0]
    else:
        return np.nan

def get_R_outer(phi, v0, r0, nu, R):
    try:
        return np.array([_get_R_outer(phi_, v0, r0, nu, R) for phi_ in phi])
    except TypeError:
        return _get_R_outer(phi, v0, r0, nu, R)

