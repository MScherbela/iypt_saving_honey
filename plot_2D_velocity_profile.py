import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from solver import get_velocity, get_R_outer, get_flow

omega = 2 * np.pi * 1
r0 = 0.5e-2
nu = 6.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
R0 = 1.449 * r0

# omega = 2 * np.pi * 1
# r0 = 0.5e-2
# nu = 10.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
# R0 = 1.56 * r0


v0 = omega * r0

n_outer = 500
phi_values_outer = np.linspace(0, 2*np.pi, n_outer)
R_outer = get_R_outer(phi_values_outer, v0, r0, nu, R0)
x_outer, y_outer = R_outer * np.cos(phi_values_outer), R_outer * np.sin(phi_values_outer)

plt.close("all")
fig, axes = plt.subplots(1,3, figsize=(12,5.6), dpi=100, gridspec_kw={'width_ratios':[1.7,1,1]})
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
for r_grid in np.arange(1, 9):
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


R_outer_values = np.linspace(r0, np.max(R_outer)*1.1, 100)
flow = get_flow(np.pi/2, v0, r0, nu, R0)
for i, ind in enumerate([0, 4, 8]):
    color=f'C{i}'
    phi = phi_quiver[ind]
    r = np.linspace(r0, R_outer_quiver[ind], 100)
    v = get_velocity(r, phi, v0, r0, nu, R_outer_quiver[ind])
    flow_R = get_flow(phi, v0, r0, nu, R_outer_values)

    ls = '--' if i  == 3 else '-'
    axes[1].plot(r*1e3, v*1e3, label=f"$\\varphi$ = {phi*180/np.pi:.0f}°", ls=ls, color=color)
    axes[2].plot(R_outer_values*1e3, flow_R * 1e3, label=f"$\\varphi$ = {phi * 180 / np.pi:.0f}°", ls=ls, color=color)
    axes[2].plot([R_outer_quiver[ind]*1e3], [flow*1e3], marker='o', ls='None', color=color)

    axes[0].plot(r*np.cos(phi)*1e3, r*np.sin(phi)*1e3, color=color, ls='--', lw=2)

axes[2].axhline(flow*1e3, color='gray', ls='--')

for ax in axes[1:3]:
    ax.set_ylim([0, None])
    ax.grid(alpha=0.5)
    ax.legend()

axes[1].set_xlabel("r [mm]")
axes[1].set_ylabel("v [mm/s]")
axes[1].set_title("Flow velocity profile")

axes[2].set_xlabel("Outer radius $R$ [mm]")
axes[2].set_ylabel("Flow rate [mm^2/s]")
axes[2].set_title("Flow rate")

fig.tight_layout()
fig.savefig("/home/mscherbela/develop/iypt_saving_honey/plots/flow_profile.png", dpi=400, bbox_inches='tight')

