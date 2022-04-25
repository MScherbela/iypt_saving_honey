import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from solver import get_velocity_wrong, get_R_outer

nu = 4.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l

phi_circle = np.linspace(0, 2*np.pi, 200)
omega = 2 * np.pi * 1.0
r0 = 0.5e-2
v0 = omega * r0

n_R0_values = 7
R0_values = np.linspace(1.1, 1.901, n_R0_values) * r0

plt.close("all")
fig, axes = plt.subplots(1,2, dpi=100, figsize=(14,7))

r_plot = np.linspace(1, 2.5, 500) *r0
axes[1].plot(r_plot * 1e3, get_velocity_wrong(r_plot, 0.0, v0, r0, nu) * 1e3, label='$\\varphi=0$: Right (dripping) side', color='k')
axes[1].plot(r_plot * 1e3, get_velocity_wrong(r_plot, np.pi / 2, v0, r0, nu) * 1e3, label='$\\varphi=90^\\circ$ Top', color='gray', ls='--')
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
    R_phi = np.array([get_R_outer(phi_, v0, r0, nu, R0) for phi_ in phi_circle])
    x, y = R_phi * np.cos(phi_circle), R_phi * np.sin(phi_circle)
    axes[0].plot(x*1e3, y*1e3, color=color, label=f"$R_0$ = {R0*1e3:.1f} mm")

    R_left_and_top = np.array([np.max(R_phi), R0])
    velocities = get_velocity_wrong(R_left_and_top, np.array([0, np.pi / 2]), v0, r0, nu)
    axes[1].plot(R_left_and_top*1e3, velocities*1e3, ls='None', marker='o', color=color)
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
axes[0].axis("equal")
axes[0].grid(alpha=0.5)
axes[0].legend(loc='upper right')
fig.suptitle(f"Varying amount of honey\nRotation speed = {omega/(2*np.pi):.1f} Hz")
fig.savefig("plots/sweep_amount_of_honey.png", dpi=300, bbox_inches='tight')

