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
N = 200
r_kpc = np.linspace(0.5, 20, N)
r = r_kpc * kpc
dr = r[1] - r[0]

# ================= MASS MODEL =================
M = 1e11 * 1.989e30
sigma = 4.5 * kpc
M_enc = M*(1 - np.exp(-r/(3*sigma)))

rho0 = np.exp(-r_kpc/4)

v0 = np.where(r_kpc < 6, 180*np.sqrt(r_kpc/2), 230)*1000

# ================= GRAVITY =================
def gN():
    return G*M_enc/(r**2 + 1e-12)

# ================= MAG FIELD =================
def Bfield(I0):
    Jz = (I0/(2*np.pi*sigma**2))*np.exp(-(r/sigma)**2)
    Ienc = np.cumsum(2*np.pi*r*Jz*dr)
    return mu0*Ienc/(2*np.pi*r + 1e-20)

# ================= ALADIN FORCE =================
def aladin_force(rho, vr, vth, p, B, alphaA):

    E_mag = B**2/(2*mu0)
    E_kin = 0.5*rho*(vr**2 + vth**2)
    E_th = p/(gamma-1)

    ratio = E_mag/(E_mag + E_kin + E_th + 1e-30)

    return alphaA*np.gradient(ratio, r)

# ================= SIMULATION CORE =================
def run_sim(alphaA, I0):

    def rhs(t, y):

        rho = y[0:N]
        vr = y[N:2*N]
        vth = y[2*N:3*N]
        p = y[3*N:4*N]

        B = Bfield(I0)

        dB = np.gradient(B, r)
        JxB = -(B/mu0)*(dB + B/(r+1e-20))

        A = aladin_force(rho, vr, vth, p, B, alphaA)

        g = gN()

        ar = JxB + A - g

        return np.concatenate([
            np.zeros_like(rho),  # simplified stability focus
            ar - vth**2/r,
            -vr*vth/(r+1e-20),
            np.zeros_like(p)
        ])

    y0 = np.concatenate([
        rho0,
        np.zeros_like(r),
        v0,
        rho0*1e10
    ])

    sol = solve_ivp(rhs, [0, 5], y0, method='LSODA')

    v_final = sol.y[2*N:3*N,-1]/1000

    # metric: flatness score (low slope = flat curve)
    slope = np.mean(np.abs(np.gradient(v_final)))

    return slope

# ================= PARAMETER SWEEP =================
alphas = np.linspace(0.0, 0.2, 8)
currents = np.linspace(1e18, 5e20, 8)

Z = np.zeros((len(alphas), len(currents)))

for i,a in enumerate(alphas):
    for j,I in enumerate(currents):
        Z[i,j] = run_sim(a, I)

# ================= PLOT PHASE DIAGRAM =================
plt.figure(figsize=(8,6))
plt.imshow(Z, origin="lower", aspect="auto",
           extent=[1e18,5e20,0,0.2],
           cmap="inferno")

plt.colorbar(label="Rotation Curve Slope (lower = flatter)")
plt.xlabel("Current I_total")
plt.ylabel("Alpha_A (ALADIN coupling)")
plt.title("ALADIN v0.4 Phase Diagram (Regime Map)")
plt.show()
