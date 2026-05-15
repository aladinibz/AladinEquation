import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "aladin_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== CONSTANTS =====================
G = 6.67430e-11
kpc = 3.086e19

# ===================== RADIUS =====================
r_kpc = np.linspace(0.1, 25, 300)
r = r_kpc * kpc

# ===================== MASS MODEL =====================
M = 1e11 * 1.989e30
sigma = 4.5 * kpc

M_enc = M * (1 - np.exp(-r/(3*sigma)))

# ===================== PURE NEWTON ROTATION CURVE =====================
v = np.sqrt(G * M_enc / (r + 1e-12)) / 1000  # km/s

# ===================== PLOT =====================
plt.figure(figsize=(8,6))
plt.plot(r_kpc, v, color="white")

plt.title("Baseline Rotation Curve (Newtonian)")
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.grid(alpha=0.3)

path = os.path.join(OUT_DIR, "baseline.png")
plt.savefig(path, dpi=200)

print("[OK] Saved:", path)

plt.show()
