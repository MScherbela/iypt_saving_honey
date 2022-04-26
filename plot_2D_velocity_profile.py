import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from solver import get_velocity, get_R_outer

omega = 2 * np.pi * 1
r0 = 0.5e-2
v0 = omega * r0
nu = 6.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
R0 = 1.449 * r0
# R0 = 1.44 * r0

n_outer = 500
phi_values_outer = np.linspace(0, 2*np.pi, n_outer)
R_outer = get_R_outer(phi_values_outer, v0, r0, nu, R0)
x_outer, y_outer = R_outer * np.cos(phi_values_outer), R_outer * np.sin(phi_values_outer)

plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(13,6), dpi=100)
axes[0].add_patch(plt.Polygon(1e3*np.array([x_outer, y_outer]).T, color='gold', alpha=0.3, zorder=-1))
axes[0].add_patch(plt.Circle((0, 0), r0 * 1e3, color='gray'))
axes[0].add_patch(matplotlib.patches.FancyArrowPatch((1e3*r0/2, 0), (0, 1e3*r0/2),
                             connectionstyle=f"arc3,rad={r0*1e3/8:f}", color='k',
                                                     arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8"))

axes[0].plot(x_outer*1e3, y_outer*1e3, lw=3, color='gold')
axes[0].axis("equal")
axes[0].set_xlabel("x / mm")
axes[0].set_xlabel("y / mm")
axes[0].set_title("Flow profile")


phi_grid = np.linspace(0, 2*np.pi, 200)
for r_grid in np.arange(1, 10):
    axes[0].plot(r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), color='gray', alpha=0.1)

n_phi_quiver = 16
n_r_quiver = 10
phi_quiver = (np.arange(n_phi_quiver) * 2 * np.pi / n_phi_quiver)
R_outer_quiver = get_R_outer(phi_quiver.flatten(), v0, r0, nu, R0)
r_quiver = np.linspace(r0, np.max(R_outer), n_r_quiver)
v_quiver = [get_velocity(r_quiver, phi, v0, r0, nu, R) for phi, R in zip(phi_quiver, R_outer_quiver)]
v_quiver = np.array(v_quiver).T
v_quiver[r_quiver[:,None] > R_outer_quiver[None, :]] = np.nan
vx_quiver = -np.sin(phi_quiver) * v_quiver
vy_quiver = np.cos(phi_quiver) * v_quiver
x_quiver, y_quiver = r_quiver[:,None] * np.cos(phi_quiver), r_quiver[:,None] * np.sin(phi_quiver)
axes[0].quiver(x_quiver*1e3, y_quiver*1e3, vx_quiver*1e3, vy_quiver*1e3, scale=0.7e3, headwidth=3, headlength=4, width=5e-3, color='brown')

for i, ind in enumerate([0, 4, 8, 12]):
    r = np.linspace(r0, R_outer_quiver[ind], 100)
    v = get_velocity(r, phi_quiver[ind], v0, r0, nu, R_outer_quiver[ind])
    ls = '--' if i  == 3 else '-'
    axes[1].plot(r*1e3, v*1e3, label=f"$\\varphi$ = {phi_quiver[ind]*180/np.pi:.0f}°", ls=ls)
axes[1].set_ylim([0, None])
axes[1].grid(alpha=0.5)
axes[1].set_xlabel("r [mm]")
axes[1].set_ylabel("v [mm/s]")
axes[1].set_title("Velocity")
axes[1].legend()

fig.tight_layout()
fig.savefig("/home/mscherbela/develop/iypt_saving_honey/plots/flow_profile.png", dpi=400, bbox_inches='tight')

