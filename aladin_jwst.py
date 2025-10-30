---

### **2. `aladin_jwst.py` — JWST SIM**

```python
# aladin_jwst.py
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 300, 1000)
M_dm = 1e5 * np.exp(t/120)
torque = 1 + 0.1 * (t > 50)
M_plasma = 1e7 * (t/80)**3 * torque
M_total = M_dm * np.sqrt(1 + 1.1 / (M_dm**2 / (t+1))) * np.exp(-t/80) + M_plasma

plt.figure(figsize=(8,6))
plt.loglog(t, M_total, 'purple', lw=3, label='Aladin Total')
plt.axhline(1e8, color='orange', ls='--', label='JADES-GS-z14-0')
plt.axvline(80, color='cyan', ls=':', label='τ_A = 80 Myr')
plt.xlabel('Time (Myr)'); plt.ylabel('Mass (M⊙)')
plt.title('ALADIN: JWST z=14 Galaxy in 80 Myr')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig('plots/jwst_growth.png', dpi=200, bbox_inches='tight')
plt.show()