import numpy as np
import matplotlib.pyplot as plt

print("🌌 v0.2.6 Galactic Rotation Curves + Turbulent KE")

# ====================== SETUP ======================
N = 250
r_max_kpc = 30.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * 3.086e19
dr = r[1] - r[0]

mu0 = 4 * np.pi * 1e-7
G = 6.6743e-11

I_total = 8e21
M_bary_total = 6e10 * 1.989e30
alpha_A = 0.15
sigma_kpc = 5.5
sigma = sigma_kpc * 3.086e19

J_z = (I_total / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2*sigma**2))
rho = 2e-21 * np.exp(-r_kpc / 5.0)

M_enc_bary = M_bary_total * (1 - np.exp(-r / (3*sigma)))

def gravity_acc(r):
    return G * M_enc_bary / (r**2 + 1e10)

I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
B = mu0 * I_enc / (2 * np.pi * r + 1e-20)
dB_dr = np.gradient(B, r)
JxB_r = - (B / mu0) * (dB_dr + B / (r + 1e-20))

dp_dr = np.gradient(1e8 * rho, r)
boost = np.clip(np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30), 0, 8)
aladin_factor = 1.0 + alpha_A * boost

a_bary = gravity_acc(r)
a_plasma_extra = -JxB_r * aladin_factor - dp_dr / (rho + 1e-30)

# DM NFW
M_vir = 1.2e12 * 1.989e30
r_s = 20 * 3.086e19
def nfw_mass(r):
    x = r / r_s
    return M_vir * (np.log(1 + x) - x/(1 + x)) / (np.log(11) - 10/11)
a_dm = G * nfw_mass(r) / (r**2 + 1e10)

v_bary = np.sqrt(np.maximum(r * a_bary, 0)) / 1000
v_plasma = np.sqrt(np.maximum(r * (a_bary + a_plasma_extra), 0)) / 1000
v_dm = np.sqrt(np.maximum(r * (a_bary + a_dm), 0)) / 1000
v_total = np.sqrt(np.maximum(r * (a_bary + a_plasma_extra + a_dm), 0)) / 1000

# ====================== KINETIC ENERGY ======================
vol = 2 * np.pi * r * dr

# Ordered rotational KE
E_rot_plasma = np.sum(0.5 * rho * (v_plasma * 1000)**2 * vol)
E_rot_dm = np.sum(0.5 * rho * (v_dm * 1000)**2 * vol)

# Turbulent KE estimation (velocity dispersion)
# Typical galactic turbulent velocity ~ 10-50 km/s (higher in star-forming regions)
sigma_turb = 25 * 1000  # m/s — reasonable average
E_turb = np.sum(0.5 * rho * 3 * (sigma_turb)**2 * vol)   # 3D isotropic turbulence

print(f"\n=== KINETIC ENERGY ===")
print(f"Ordered Rotational KE (Plasma) : {E_rot_plasma:.2e} J")
print(f"Ordered Rotational KE (DM)     : {E_rot_dm:.2e} J")
print(f"Estimated Turbulent KE         : {E_turb:.2e} J")
print(f"Turbulent / Rotational (Plasma): {E_turb / E_rot_plasma:.3f}")
print(f"Peak B-field                   : {B.max()*1e6:.2f} μG")

# Plot
plt.style.use('dark_background')
plt.figure(figsize=(13, 8))

plt.plot(r_kpc, v_bary, 'white', lw=2, label='Baryons only')
plt.plot(r_kpc, v_plasma, 'cyan', lw=3, label='Baryons + Plasma (Aladin)')
plt.plot(r_kpc, v_dm, 'orange', lw=2.5, ls='--', label='Baryons + DM (NFW)')
plt.plot(r_kpc, v_total, 'yellow', lw=2.5, label='Combined')
plt.axhline(230, color='red', ls=':', label='Milky Way ~230 km/s')

plt.title('v0.2.6 Rotation Curves + Turbulent KE')
plt.xlabel('Radius [kpc]')
plt.ylabel('Velocity [km/s]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
