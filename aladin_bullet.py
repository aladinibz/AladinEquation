# aladin_bullet.py — ADD TO REPO
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
dm = np.exp(-x**2 / 0.5**2)
plasma = 0.8 * np.exp(-(x-1.3)**2 / 0.3**2) + 0.2 * np.exp(-(x+1.3)**2 / 0.3**2)
total = dm + plasma

plt.plot(x, dm, 'b--', label='DM Halo')
plt.plot(x, plasma, 'r-', label='Plasma Shear')
plt.plot(x, total, 'purple', lw=3, label='Aladin Total')
plt.axvline(1.3, color='orange', ls=':', label='1.3 Mpc Offset')
plt.xlabel('Distance (Mpc)'); plt.ylabel('Mass Density')
plt.title('Bullet Cluster: 1.3 Mpc Offset')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('plots/bullet_offset.png', dpi=200)
plt.show()
