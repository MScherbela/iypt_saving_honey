import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from scipy.optimize import root, brentq

g = 9.81
nu = 4.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l


def get_velocity(r, phi, v0, r0, g, nu):
    """Calculate the velocity profile, i.e. the flow velocity in tangential direction for an arbitrary r and phi.
    i.e. the solution of the navier-stokes equation"""
    s2 = np.sqrt(2)
    x = r / r0
    return v0 * x + ((1+s2)/2 * x**s2 + (1-s2)/2 / x**s2 - x**2) * (g*r0**2)/(2*nu) * np.cos(phi)

def _flow_root_func(x, phi, v0, r0, g, nu, R0):
    """Helper function to determine the maximum radius at each phi"""
    flow = v0 * r0 * ((R0 / r0) ** 2 - 1) / 2
    A = v0 * r0 * (x**2 - 1) / 2
    s2 = np.sqrt(2)
    B = (0.5 * x**(1+s2) + 0.5 * x**(1-s2) - (x**3)/3 - 2/3)
    C = (g*r0**3) / (2*nu) * np.cos(phi)
    return A + B*C - flow

def get_R_outer(phi, v0, r0, g, nu, R0):
    """Calculate the outer perimeter of the honey-blob as a function of phi, using conservation of flow"""
    res = root(_flow_root_func, 1.0, args=(phi, v0, r0, g, nu, R0))
    if res.success:
        return res.x[0] * r0
    else:
        return np.nan


phi_circle = np.linspace(0, 2*np.pi, 200)
omega = 2 * np.pi * 1.0
r0 = 0.5e-2
v0 = omega * r0

n_R0_values = 7
R0_values = np.linspace(1.1, 1.901, n_R0_values) * r0

plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(14,7))

r_plot = np.linspace(1, 2.5, 500) *r0
axes[1].plot(r_plot*1e3, get_velocity(r_plot, 0.0, v0, r0, g, nu)*1e3, label='$\\varphi=0$: Right (dripping) side', color='k')
axes[1].plot(r_plot*1e3, get_velocity(r_plot, np.pi/2, v0, r0, g, nu)*1e3, label='$\\varphi=90^\\circ$ Top', color='gray', ls='--')
axes[1].legend()
axes[1].set_xlabel("Radius $r$ [mm]")
axes[1].set_ylabel("Flow velocity $v$ [mm/s]")
axes[1].grid(alpha=0.5)

axes[0].add_patch(plt.Circle((0, 0), r0 * 1e3, color='gray'))
axes[0].add_patch(matplotlib.patches.FancyArrowPatch((1e3*r0/2, 0), (0, 1e3*r0/2),
                             connectionstyle=f"arc3,rad={r0*1e3/8:f}", color='k',
                                                     arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8"))

for ind_R0, R0 in enumerate(R0_values):
    color = f'C{ind_R0}'
    R_phi = np.array([get_R_outer(phi_, v0, r0, g, nu, R0) for phi_ in phi_circle])
    x, y = R_phi * np.cos(phi_circle), R_phi * np.sin(phi_circle)
    axes[0].plot(x*1e3, y*1e3, color=color, label=f"$R_0$ = {R0*1e3:.1f} mm")

    R_left_and_top = np.array([np.max(R_phi), R0])
    velocities = get_velocity(R_left_and_top, np.array([0, np.pi/2]), v0, r0, g, nu)
    axes[1].plot(R_left_and_top*1e3, velocities*1e3, ls='None', marker='o', color=color)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].axis("equal")
axes[0].grid(alpha=0.5)
axes[0].legend(loc='upper right')
fig.suptitle(f"Varying amount of honey\nRotation speed = {omega/(2*np.pi):.1f} Hz")
fig.savefig("plots/sweep_amount_of_honey.png", dpi=300, bbox_inches='tight')

