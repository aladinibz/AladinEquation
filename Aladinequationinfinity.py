import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from scipy.integrate import solve_ivp

# ===================== BACKEND SAFE MODE =====================
# Works in scripts + Colab + terminals
try:
    import IPython
    IN_COLAB = True
except:
    IN_COLAB = False

if not IN_COLAB:
    matplotlib.use("Agg")  # safe for .py execution (no GUI issues)

# ===================== OUTPUT FOLDER =====================
out_dir = "aladin_results"
os.makedirs(out_dir, exist_ok=True)

# ===================== CONSTANTS =====================
G = 6.67430e-11
mu0 = 4*np.pi*1e-7
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

M_enc = M*(1 - np.exp(-r/(3*sigma)))

rho0 = 2e-20*np.exp(-r_kpc/4)

v_theta0 = np.where(r_kpc < 6,
                    180*np.sqrt(r_kpc/2),
                    230)*1000

# ===================== GRAVITY =====================
def g_newton():
    return G*M_enc/(r**2 + 1e-12)

# ===================== MAG FIELD =====================
I0 = 3e19
Jz = (I0/(2*np.pi*sigma**2))*np.exp(-(r/sigma)**2)

def Bfield():
    Ienc = np.cumsum(2*np.pi*r*Jz*dr)
    return mu0*Ienc/(2*np.pi*r + 1e-20)

# ===================== ALADIN FORCE =====================
alphaA = 0.08

def aladin_force(rho, vr, vth, p, B):

    E_mag = B**2/(2*mu0)
    E_kin = 0.5*rho*(vr**2 + vth**2)
    E_th = p/(gamma-1)

    ratio = E_mag/(E_mag + E_kin + E_th + 1e-30)

    return alphaA*np.gradient(ratio, r)

# ===================== SYSTEM =====================
def rhs(t, y):

    rho = y[0:N]
    vr = y[N:2*N]
    vth = y[2*N:3*N]
    p = y[3*N:4*N]

    B = Bfield()

    dB = np.gradient(B, r)
    JxB = -(B/mu0)*(dB + B/(r+1e-20))

    A = aladin_force(rho, vr, vth, p, B)

    g = g_newton()

    ar = JxB + A - g

    dvr = ar - vth**2/r
    dvth = -(vr*vth/(r+1e-20))

    drho = -rho*(1/r)*np.gradient(r*vr, r)
    dpdt = -gamma*p*(1/r)*np.gradient(r*vr, r)

    return np.concatenate([drho, dvr, dvth, dpdt])

# ===================== INITIAL STATE =====================
y0 = np.concatenate([
    rho0,
    np.zeros_like(r),
    v_theta0,
    rho0*1e10
])

# ===================== RUN =====================
sol = solve_ivp(rhs, [0, 8], y0, method='LSODA')

# ===================== RESULTS =====================
v_final = sol.y[2*N:3*N,-1]/1000

# ===================== PLOT =====================
plt.figure(figsize=(8,6))
plt.plot(r_kpc, v_final, color="cyan")
plt.title("ALADIN v0.4 Universal (Colab + Python)")
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.grid(alpha=0.3)

# save always
plt.savefig(os.path.join(out_dir, "rotation_curve.png"), dpi=200)

# show only if interactive
if IN_COLAB:
    plt.show()
else:
    print("Plot saved in:", out_dir)

print("ALADIN simulation complete")
