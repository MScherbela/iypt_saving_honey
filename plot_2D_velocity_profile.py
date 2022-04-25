import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from solver import get_velocity_wrong, get_R_outer

omega = 2 * np.pi * 1.0
r0 = 0.5e-2
v0 = omega * r0
nu = 4.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
R0 = 1.9 * r0

phi_values_outer = np.linspace(0, 2*np.pi, 200)
R_outer = get_R_outer(phi_values_outer, v0, r0, nu, R0)
x_outer, y_outer = R_outer * np.cos(phi_values_outer), R_outer * np.sin(phi_values_outer)

plt.close("all")
fig, ax = plt.subplots(1,1, figsize=(14,8), dpi=100)
ax.add_patch(plt.Circle((0, 0), r0 * 1e3, color='gray'))
ax.add_patch(matplotlib.patches.FancyArrowPatch((1e3*r0/2, 0), (0, 1e3*r0/2),
                             connectionstyle=f"arc3,rad={r0*1e3/8:f}", color='k',
                                                     arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8"))

ax.plot(x_outer*1e3, y_outer*1e3, lw=3, color='gold')
ax.axis("equal")
ax.grid(alpha=0.5, zorder=-1)

n_phi_quiver = 16
n_r_quiver = 10
phi_quiver = (np.arange(n_phi_quiver) * 2 * np.pi / n_phi_quiver)[None, :]
r_quiver = np.linspace(r0, np.max(R_outer), n_r_quiver)[:, None]
R_outer_quiver = get_R_outer(phi_quiver.flatten(), v0, r0, nu, R0)[None, :]

v_quiver = get_velocity_wrong(r_quiver, phi_quiver, v0, r0, nu)
v_quiver[r_quiver > R_outer_quiver] = np.nan
vx_quiver = -np.sin(phi_quiver) * v_quiver
vy_quiver = np.cos(phi_quiver) * v_quiver
x_quiver, y_quiver = r_quiver * np.cos(phi_quiver), r_quiver * np.sin(phi_quiver)
ax.quiver(x_quiver*1e3, y_quiver*1e3, vx_quiver*1e3, vy_quiver*1e3, scale=1e3, headwidth=3, headlength=4, width=3e-3, color='brown')



