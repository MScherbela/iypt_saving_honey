import numpy as np
import matplotlib.pyplot as plt
from solver import get_velocity

nu = 4.0 / 1400 # wikipedia: mu = 2000 - 10000 mPas; density = 1.34 - 1.45 kg/l
omega = 2 * np.pi * 1.0
r0 = 0.5e-2
v0 = omega * r0
R0 = 1.6 * r0

phi = 0
r = np.linspace(r0, R0, 300)


plt.close("all")
for phi in np.array([0, 90, 180]) * np.pi/180:
    v = get_velocity(r, phi, v0, r0, nu, R0)
    plt.plot(r*1e3, v*1e3, label=f"$\\varphi$={phi*180/np.pi:.0f}")
plt.plot([r0*1e3, R0*1e3], [v0*1e3, v0*R0/r0*1e3], ls='--', color='gray')
plt.grid(alpha=0.5)
plt.ylim([-100,200])
plt.axhline(0, color='k', lw=2)
plt.axhline(v0*1e3, color='gray', lw=1)

plt.axvline(r0*1e3, color='gray', lw=1)
plt.axvline(R0*1e3, color='gray', lw=1)
plt.xlim([0, None])
plt.legend()


