import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp

# ===================== AUTO MODE DETECTION =====================
try:
    from IPython import get_ipython
    IN_COLAB = get_ipython() is not None
except:
    IN_COLAB = False

# ===================== BACKEND FIX =====================
if not IN_COLAB:
    matplotlib.use("Agg")

# ===================== OUTPUT FOLDER =====================
OUT_DIR = "aladin_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== CONSTANTS =====================
G = 6.67430e-11
gamma = 5/3
kpc = 3.086e19

# ===================== GRID =====================
N = 300
r_kpc = np.linspace(0.1, 25, N)
r = r_kpc * kpc
dr = r[1] - r[0]

# ===================== GALAXY MODEL =====================
M = 1e11 * 1.989e30
sigma = 4.5 * kpc

M_enc = M * (1 - np.exp(-r/(3*sigma)))

rho0 = 2e-20 * np.exp(-r_kpc/4)

v_theta0 = np.where(
    r_kpc < 6,
    180 * np.sqrt(r_kpc/2),
    230
) * 1000

# ===================== NEWTON GRAVITY ONLY =====================
def g_newton():
    return G * M_enc / (r**2 + 1e-12)

# ===================== SYSTEM (NO ALADIN, NO MAG FORCE) =====================
def rhs(t, y):

    rho = y[0:N]
    vr = y[N:2*N]
    vth = y[2*N:3*N]
    p = y[3*N:4*N]

    g = g_newton()

    # Pure gravity + pressure support only
    ar = -g

    dvr = ar - vth**2 / r
    dvth = -(vr * vth / (r + 1e-20))

    divv = (1/r) * np.gradient(r * vr, r)

    drho = -rho * divv - vr * np.gradient(rho, r)
    dpdt = -gamma * p * divv

    return np.concatenate([drho, dvr, dvth, dpdt])

# ===================== INITIAL CONDITIONS =====================
rho0 = np.clip(rho0, 1e-30, None)

y0 = np.concatenate([
    rho0,
    np.zeros_like(r),
    v_theta0,
    rho0 * 1e10
])

# ===================== RUN =====================
sol = solve_ivp(rhs, [0, 8], y0, method="LSODA")

# ===================== RESULT =====================
v_final = sol.y[2*N:3*N, -1] / 1000

# ===================== PLOT =====================
plt.figure(figsize=(8,6))
plt.plot(r_kpc, v_final, color="white", label="NO ALADIN (baseline)")

plt.title("Baseline Galaxy Rotation Curve (Newton Only)")
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.grid(alpha=0.3)
plt.legend()

# save ALWAYS
path = os.path.join(OUT_DIR, "baseline_rotation_curve.png")
plt.savefig(path, dpi=200)

print("[BASELINE] Saved:", path)

# show in colab only
if IN_COLAB:
    plt.show()
else:
    plt.close()
