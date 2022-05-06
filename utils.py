import numpy as np
import matplotlib.pyplot as plt

def draw_radial_grid(r, ax=None):
    ax = ax or plt.gca()
    phi = np.linspace(0, 2*np.pi, 200)
    for r_ in r:
        ax.plot(r_*np.cos(phi), r_*np.sin(phi), color='gray', alpha=0.5, zorder=-1)