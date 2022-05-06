import numpy as np
import matplotlib.pyplot as plt
from solver import get_flow
from utils import draw_radial_grid

omega = 2 * np.pi * 1
r0 = 0.5e-2
nu = 6.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
R0 = 1.2 * r0
v0 = omega * r0

n_phi = 200
phi = np.arange(n_phi) * 2 * np.pi / n_phi
dphi = (2*np.pi) / n_phi

dt = 0.2e-4
n_t = 120_000


R_outer = np.zeros([n_t, n_phi])
R_outer[0] = np.ones(n_phi) * R0
R_outer[0] += r0 * np.exp(-((phi-np.pi/2)/0.5)**2) * 0.1
flow = np.zeros([n_t, n_phi])
flow[0] = get_flow(phi, v0, r0, nu, R_outer[0])

for ind_t in range(n_t-1):
    delta_flow = 0.5 * (np.roll(flow[ind_t], 1) - np.roll(flow[ind_t], -1)) / dphi
    dR = delta_flow * dt / R_outer[ind_t]
    R_outer[ind_t+1] = R_outer[ind_t] + dR
    flow[ind_t+1] = get_flow(phi, v0, r0, nu, R_outer[ind_t+1])
#%%
plt.close("all")
fig, axes = plt.subplots(1,2, figsize=(14,8), dpi=100)

line = axes[1].plot(phi, R_outer[ind_t, :])[0]
axes[1].set_ylim([r0*1e3, 1.6*r0*1e3])

def draw_R_outer(phi, R, line=None, ax=None):
    phi = np.concatenate([phi, phi[:1]])
    R = np.concatenate([R, R[:1]])
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    if line is None:
        ax = ax or plt.gca()
        line = ax.plot(x,y)[0]
    line.set_data(x,y)
    return line

def draw_cross(phi, R, line=None, ax=None):
    phi_values = np.array([phi, phi-np.pi, np.nan, phi-np.pi/2, phi+np.pi/2])
    x = R * np.cos(phi_values)
    y = R * np.sin(phi_values)

    if line is None:
        ax = ax or plt.gca()
        line = ax.plot(x,y, color='brown')[0]
    line.set_data(x,y)
    return line

line_R_outer = draw_R_outer(phi, R_outer[0], ax=axes[0])
line_cross = draw_cross(0, r0*1e3, ax=axes[0])
axes[0].axis("equal")
draw_radial_grid(np.arange(r0, 1.6*r0, 5e-4)*1e3, ax=axes[0])

for ind_t in np.arange(0, n_t, 100):
    draw_R_outer(phi, R_outer[ind_t]*1e3, line_R_outer)
    draw_cross(ind_t*dt*omega, r0*1e3, line_cross)

    line.set_data(phi, R_outer[ind_t]*1e3)
    plt.pause(0.02)





