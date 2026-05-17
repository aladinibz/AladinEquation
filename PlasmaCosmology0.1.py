import numpy as np
import matplotlib.pyplot as plt

print("🌌 v0.3.0 Galactic Rotation Curves + Cosmic Ray Pressure")

# ====================== SETUP ======================
N = 250
r_max_kpc = 30.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * 3.086e19
dr = r[1] - r[0]

mu0 = 4 * np.pi * 1e-7
G = 6.6743e-11

h_kpc = 0.4
h = h_kpc * 3.086e19
vol = 2 * np.pi * r * dr * h

# ====================== PARAMETERS ======================
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

dp_thermal_dr = np.gradient(1e8 * rho, r)

# ====================== COSMIC RAY PRESSURE ======================
# Simple radial profile: higher in center, declining outward
P_cr = 5e-13 * np.exp(-r_kpc / 8.0)   # typical galactic CR pressure ~10^{-12} - 10^{-13} erg/cm³
dP_cr_dr = np.gradient(P_cr, r)
a_cr = - dP_cr_dr / (rho + 1e-30)

# Turbulent pressure
sigma_turb = 25 * 1000
P_turb = rho * sigma_turb**2
dP_turb_dr = np.gradient(P_turb, r)
a_turb = - dP_turb_dr / (rho + 1e-30)

boost = np.clip(np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30), 0, 8)
aladin_factor = 1.0 + alpha_A * boost

a_bary = gravity_acc(r)
a_plasma_extra = -JxB_r * aladin_factor - dp_thermal_dr / (rho + 1e-30) + a_turb + a_cr

# DM
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

# ====================== ENERGY DENSITIES ======================
u_rot_plasma = 0.5 * rho * (v_plasma * 1000)**2
u_rot_dm     = 0.5 * rho * (v_dm * 1000)**2
u_turb       = 0.5 * rho * 3 * sigma_turb**2
u_mag        = B**2 / (2 * mu0)
u_therm      = 1e8 * rho
u_cr         = P_cr   # for ultra-relativistic CRs, u_cr ≈ 3 P_cr, but we use P_cr for simplicity

print(f"Peak B-field : {B.max()*1e6:.2f} μG")
print(f"\n=== AVERAGE ENERGY DENSITIES (J/m³) ===")
print(f"Rotational (Plasma) : {np.mean(u_rot_plasma):.2e}")
print(f"Rotational (DM)     : {np.mean(u_rot_dm):.2e}")
print(f"Turbulent           : {np.mean(u_turb):.2e}")
print(f"Magnetic            : {np.mean(u_mag):.2e}")
print(f"Thermal             : {np.mean(u_therm):.2e}")
print(f"Cosmic Rays         : {np.mean(u_cr):.2e}")

# ====================== PLOTS ======================
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].plot(r_kpc, v_bary, 'white', lw=2, label='Baryons only')
axs[0].plot(r_kpc, v_plasma, 'cyan', lw=3, label='Plasma + Turb + CR')
axs[0].plot(r_kpc, v_dm, 'orange', lw=2.5, ls='--', label='DM')
axs[0].plot(r_kpc, v_total, 'yellow', lw=2.5, label='Combined')
axs[0].axhline(230, color='red', ls=':', label='Milky Way')
axs[0].set_title('Rotation Curves')
axs[0].set_xlabel('Radius [kpc]')
axs[0].set_ylabel('Velocity [km/s]')
axs[0].legend()
axs[0].grid(alpha=0.3)

# Energy Densities
labels = ['Rot (Plasma)', 'Rot (DM)', 'Turb', 'Mag', 'Therm', 'CR']
values = [np.mean(u_rot_plasma), np.mean(u_rot_dm), np.mean(u_turb), 
          np.mean(u_mag), np.mean(u_therm), np.mean(u_cr)]
axs[1].bar(labels, values, color=['cyan', 'orange', 'lime', 'magenta', 'white', 'purple'])
axs[1].set_title('Average Energy Densities')
axs[1].set_ylabel('Energy Density [J/m³]')
axs[1].tick_params(axis='x', rotation=45)

plt.suptitle('v0.3.0 Plasma Model with Cosmic Ray Pressure')
plt.tight_layout()
plt.show()
