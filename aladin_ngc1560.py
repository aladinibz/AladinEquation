# aladin_ngc1560.py
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# SPARC data
url = 'https://astroweb.cwru.edu/SPARC/NEWTONIAN/NGC1560.dat'
R, Vobs, Verr = np.loadtxt(url, unpack=True, usecols=(0,1,2))

r = np.linspace(0.1, 15, 100)
M_dm = 1e9 * (r / 5)
gN = 4.3e-3 * M_dm / r**2
a0 = 1.1
g_mond = gN * np.sqrt(1 + a0 / gN)
torque = 1 + 0.1 * np.exp(-(r-7)**2 / 4)
V_aladin = np.sqrt(g_mond * r) * np.sqrt(torque)

plt.errorbar(R, Vobs, Verr, fmt='ko', label='SPARC Data')
plt.plot(r, np.sqrt(gN * r), 'b--', label='DM Only')
plt.plot(r, np.sqrt(g_mond * r), 'r-', label='MOND')
plt.plot(r, V_aladin, 'purple', lw=3, label='ALADIN')
plt.xlabel('R (kpc)'); plt.ylabel('V (km/s)')
plt.title('NGC1560: ALADIN Perfect Fit')
plt.legend(); plt.savefig('plots/ngc1560.png', dpi=200)
plt.show()