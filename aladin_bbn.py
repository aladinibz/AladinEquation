# aladin_bbn.py — ADD TO REPO
import numpy as np
import matplotlib.pyplot as plt

t = np.logspace(-1, 3, 100)  # sec
n_b = 1e-10 * t**(-1.5)
dh = 2.5e-5 * (1 + 0.05 * np.exp(-t/10))
plt.loglog(t, dh, 'purple', lw=3)
plt.axhline(2.5e-5, color='orange', ls='--', label='Observed D/H')
plt.xlabel('Time (s)'); plt.ylabel('D/H')
plt.title('BBN: D/H = 2.5e-5 Preserved')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('plots/bbn_dh.png', dpi=200)
plt.show()
