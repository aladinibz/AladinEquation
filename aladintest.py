import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use("dark_background")

# ================= CONSTANTS =================
G = 6.67430e-11
mu0 = 4*np.pi*1e-7
gamma = 5/3
kpc = 3.086e19

# ================= GRID =================
N = 300
r_kpc = np.linspace(0.1, 25, N)
r = r_kpc * kpc
dr = r[1] - r[0]

# ================= MASS MODEL =================
M = 1e11 * 1.989e30
sigma = 4.5 * kpc
M_enc = M*(1 - np.exp(-r/(3*sigma)))

rho0 = 2e-20*np.exp(-r_kpc/4)

v_theta0 = np.where(r_kpc < 6,
                    180*np.sqrt(r_kpc/2),
                    230)*1000

# ================= GRAVITY =================
def g_newton():
    return G*M_enc/(r**2 + 1e-12)

# ================= MAG FIELD =================
I0 = 3e19
Jz = (I0/(2*np.pi*sigma**2))*np.exp(-(r/sigma)**2)

def Bfield():
    Ienc = np.cumsum(2*np.pi*r*Jz*dr)
    return mu0*Ienc/(2*np.pi*r + 1e-20)

# ================= SYSTEM (NO ALADIN) =================
def rhs(t, y):

    rho = y[0:N]
    vr = y[N:2*N]
    vth = y[2*N:3*N]
    p = y[3*N:4*N]

    B = Bfield()

    dB = np.gradient(B, r)
    JxB = -(B/mu0)*(dB + B/(r+1e-20))

    g = g_newton()

    # ONLY PHYSICS TERMS (NO ALADIN)
    ar = JxB - g

    dvr = ar - vth**2/r
    dvth = -(vr*vth/(r+1e-20))

    drho = -rho*(1/r)*np.gradient(r*vr, r)
    dpdt = -gamma*p*(1/r)*np.gradient(r*vr, r)

    return np.concatenate([drho, dvr, dvth, dpdt])

# ================= INITIAL STATE =================
y0 = np.concatenate([
    rho0,
    np.zeros_like(r),
    v_theta0,
    rho0*1e10
])

# ================= RUN =================
sol = solve_ivp(rhs, [0, 8], y0, method='LSODA')

# ================= RESULT =================
v_final = sol.y[2*N:3*N,-1]/1000

plt.figure(figsize=(8,6))
plt.plot(r_kpc, v_final, color="white", label="ALADIN OFF (baseline)")
plt.title("Baseline Rotation Curve (NO ALADIN)")
plt.xlabel("Radius (kpc)")
plt.ylabel("Velocity (km/s)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
