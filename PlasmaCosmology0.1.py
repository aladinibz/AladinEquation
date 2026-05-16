import numpy as np
import matplotlib.pyplot as plt

print("🌌 v0.2.3 Galactic Rotation Curves — Clean Force Balance + Comparisons")

# ====================== SETUP ======================
N = 250
r_max_kpc = 30.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * 3.086e19
dr = r[1] - r[0]

mu0 = 4 * np.pi * 1e-7
G = 6.6743e-11

# ====================== PARAMETERS ======================
I_total = 1.5e20                # Aiming for realistic ~few μG
M_bary_total = 6e10 * 1.989e30
alpha_A = 0.12
sigma_kpc = 5.5
sigma = sigma_kpc * 3.086e19

J_z = (I_total / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2*sigma**2))
rho = 2e-21 * np.exp(-r_kpc / 5.0)

M_enc_bary = M_bary_total * (1 - np.exp(-r / (3*sigma)))

def gravity_acc(r):
    return G * M_enc_bary / (r**2 + 1e10)

# ====================== COMPUTE FORCES ======================
I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
B = mu0 * I_enc / (2 * np.pi * r + 1e-20)

dB_dr = np.gradient(B, r)
JxB_r = - (B / mu0) * (dB_dr + B / (r + 1e-20))   # inward

dp_dr = np.gradient(1e8 * rho, r)

# Safe Aladin factor
boost = np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30)
boost = np.clip(boost, 0.0, 5.0)                    # prevent blow-up
aladin_factor = 1.0 + alpha_A * boost

a_bary = gravity_acc(r)
a_plasma_extra = -JxB_r * aladin_factor - dp_dr / (rho + 1e-30)

# ====================== DARK MATTER (NFW) ======================
M_vir = 1.2e12 * 1.989e30
r_s = 20 * 3.086e19

def nfw_mass(r):
    x = r / r_s
    return M_vir * (np.log(1 + x) - x/(1 + x)) / (np.log(11) - 10/11)

a_dm = G * nfw_mass(r) / (r**2 + 1e10)

# ====================== VELOCITIES ======================
v_bary_only   = np.sqrt(np.maximum(r * a_bary, 0)) / 1000
v_plasma_only = np.sqrt(np.maximum(r * (a_bary + a_plasma_extra), 0)) / 1000
v_dm_only     = np.sqrt(np.maximum(r * (a_bary + a_dm), 0)) / 1000
v_total       = np.sqrt(np.maximum(r * (a_bary + a_plasma_extra + a_dm), 0)) / 1000

print(f"Peak B-field          : {B.max()*1e6:.2f} μG")
print(f"Peak Plasma-only v    : {v_plasma_only.max():.1f} km/s")
print(f"Peak DM-only v        : {v_dm_only.max():.1f} km/s")
print(f"Peak Combined v       : {v_total.max():.1f} km/s")

# ====================== PLOT ======================
plt.style.use('dark_background')
plt.figure(figsize=(13, 8))

plt.plot(r_kpc, v_bary_only, 'white', lw=2, label='Baryons only')
plt.plot(r_kpc, v_plasma_only, 'cyan', lw=3, label='Baryons + Plasma (Aladin)')
plt.plot(r_kpc, v_dm_only, 'orange', lw=2.5, ls='--', label='Baryons + DM (NFW)')
plt.plot(r_kpc, v_total, 'yellow', lw=2.5, label='Plasma + DM Combined')
plt.axhline(230, color='red', ls=':', lw=2, label='Milky Way ~230 km/s')

plt.title('v0.2.3 Galactic Rotation Curves — Plasma vs Dark Matter')
plt.xlabel('Radius [kpc]')
plt.ylabel('Circular Velocity [km/s]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
