import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== COLAB FIX =====================
%matplotlib inline

# ====================== CONSTANTS ======================
mu0 = 4 * np.pi * 1e-7
gamma = 5/3
G = 6.67430e-11
kpc_to_m = 3.086e19

# ====================== GRID ======================
N = 300
r_kpc = np.linspace(0.1, 25, N)
r = r_kpc * kpc_to_m
dr = r[1] - r[0]
vol = 2 * np.pi * r * dr

# ====================== PARAMETERS ======================
alpha_A = 0.08
I_total = 3e19
M_bary_total = 8e10 * 1.989e30
sigma = 4.5 * kpc_to_m

J_z = (I_total / (2*np.pi*sigma**2)) * np.exp(-(r/sigma)**2)

rho0 = 2e-20 * np.exp(-r_kpc / 4)

v_theta0 = np.where(
    r_kpc < 6,
    180*np.sqrt(r_kpc / 2),
    230
) * 1000

M_enc = M_bary_total * (1 - np.exp(-r/(3*sigma)))

def gravity(r):
    return G * M_enc / (r**2 + 1e-12)

# ====================== MAG FIELD ======================
def Bfield():
    I_enc = np.cumsum(2*np.pi*r*J_z*dr)
    B = mu0 * I_enc / (2*np.pi*r + 1e-20)
    return B

# ====================== ALADIN OPERATOR ======================
def aladin_force(rho, vr, vth, p, B):

    E_mag = B**2 / (2 * mu0)
    E_kin = 0.5 * rho * (vr**2 + vth**2)
    E_th  = p / (gamma - 1)

    E_tot = E_mag + E_kin + E_th + 1e-30
    ratio = E_mag / E_tot

    return alpha_A * np.gradient(ratio, r)

# ====================== RHS ======================
def rhs(t, y):

    rho = y[0:N]
    vr = y[N:2*N]
    vth = y[2*N:3*N]
    p = y[3*N:4*N]

    B = Bfield()

    dB = np.gradient(B, r)
    JxB = -(B / mu0) * (dB + B / (r + 1e-20))

    A_ALADIN = aladin_force(rho, vr, vth, p, B)

    dp_dr = np.gradient(p, r)
    g = gravity(r)

    a_r = JxB + A_ALADIN - dp_dr/(rho + 1e-30) - g

    dvr = a_r - (vth**2 / r)
    dvth = -(vr * vth / (r + 1e-30))

    div_v = (1/r) * np.gradient(r * vr, r)

    drho = -rho * div_v - vr * np.gradient(rho, r)
    dpdt = -gamma * p * div_v

    return np.concatenate([drho, dvr, dvth, dpdt])

# ====================== INITIAL CONDITIONS ======================
y0 = np.concatenate([
    rho0,
    np.zeros_like(r),
    v_theta0,
    rho0 * 1e10
])

# ====================== SOLVE ======================
sol = solve_ivp(rhs, [0, 8], y0, method='LSODA')

# ====================== PLOT ======================
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    r_kpc,
    sol.y[2*N:3*N, -1] / 1000,
    color="cyan",
    lw=2
)

ax.set_title("ALADIN v0.2 Rotation Curve")
ax.set_xlabel("Radius (kpc)")
ax.set_ylabel("Velocity (km/s)")
ax.grid(alpha=0.3)

plt.show()
