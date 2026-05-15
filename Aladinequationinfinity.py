import numpy as np
import matplotlib.pyplot as plt

# ================= CONSTANTS =================
G = 6.67430e-11
a0 = 1.2e-10

# ================= GRID =================
r = np.linspace(0.5, 30, 400)

# ================= BASE MASS PROFILE =================
M = 1e11 * 1.989e30

gN = G * M / r**2

# ================= MOND LAYER =================
def mond(gN):
    return np.sqrt(gN * a0)

# ================= PLASMA LAYER (synthetic) =================
def plasma_term(r):
    return 0.3 * np.exp(-r/10) * np.sin(r)

# ================= ALADIN CORE HYPOTHESIS =================
def aladin_total(alphaA, t):

    # MOND transition
    g_mond = mond(gN)

    # plasma contribution
    g_plasma = plasma_term(r)

    # time evolution (your GeniePower simplified)
    genie = (np.log(1+t) + np.sin(t) + np.exp(-t/2))

    return (g_mond + alphaA * g_plasma) * genie

# ================= RUN TESTS =================
t_vals = [0.1, 1, 5, 10]

plt.style.use("dark_background")

for t in t_vals:
    g = aladin_total(0.1, t)
    v = np.sqrt(g * r)

    plt.plot(r, v, label=f"t={t}")

plt.title("ALADIN v0.2: Rotation Curve Evolution Test")
plt.xlabel("Radius")
plt.ylabel("Velocity")
plt.legend()
plt.show()
