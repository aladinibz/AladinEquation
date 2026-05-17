import numpy as np
import matplotlib.pyplot as plt

print("🌌 v0.4.1 Galactic Rotation Curves + Radial Energy Density Ratios")

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
rho = 1.8e-20 * np.exp(-r_kpc / 5.5)

M_enc_bary = M_bary_total * (1 - np.exp(-r / (3*sigma)))

def gravity_acc(r):
    return G * M_enc_bary / (r**2 + 1e10)

I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
B = mu0 * I_enc / (2 * np.pi * r + 1e-20)
dB_dr = np.gradient(B, r)
JxB_r = - (B / mu0) * (dB_dr + B / (r + 1e-20))

dp_dr = np.gradient(1e8 * rho, r)

# Turb + CR
sigma_turb = 25 * 1000
P_turb = rho * sigma_turb**2
a_turb = -np.gradient(P_turb, r) / (rho + 1e-30)

P_cr = 5e-13 * np.exp(-r_kpc / 8.0)
a_cr = -np.gradient(P_cr, r) / (rho + 1e-30)

boost = np.clip(np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30), 0, 8)
aladin_factor = 1.0 + alpha_A * boost

a_bary = gravity_acc(r)
a_plasma_extra = -JxB_r * aladin_factor - dp_dr / (rho + 1e-30) + a_turb + a_cr

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

# ====================== RADIAL ENERGY DENSITIES ======================
u_rot_plasma = 0.5 * rho * (v_plasma * 1000)**2
u_rot_dm     = 0.5 * rho * (v_dm * 1000)**2
u_turb       = 0.5 * rho * 3 * sigma_turb**2
u_mag        = B**2 / (2 * mu0)
u_therm      = 1e8 * rho
u_cr         = P_cr

print(f"Peak B-field : {B.max()*1e6:.2f} μG")

# ====================== PLOTS ======================
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 1. Rotation Curves
axs[0,0].plot(r_kpc, v_bary, 'white', lw=2, label='Baryons only')
axs[0,0].plot(r_kpc, v_plasma, 'cyan', lw=3, label='Plasma + Turb + CR')
axs[0,0].plot(r_kpc, v_dm, 'orange', lw=2.5, ls='--', label='DM')
axs[0,0].plot(r_kpc, v_total, 'yellow', lw=2.5, label='Combined')
axs[0,0].axhline(230, color='red', ls=':', label='Milky Way')
axs[0,0].set_title('Rotation Curves')
axs[0,0].set_xlabel('Radius [kpc]')
axs[0,0].set_ylabel('Velocity [km/s]')
axs[0,0].legend()
axs[0,0].grid(alpha=0.3)

# 2. Radial Energy Densities
axs[0,1].plot(r_kpc, u_rot_plasma, 'cyan', lw=2, label='Rot (Plasma)')
axs[0,1].plot(r_kpc, u_rot_dm, 'orange', lw=2, label='Rot (DM)')
axs[0,1].plot(r_kpc, u_turb, 'lime', lw=2, label='Turb')
axs[0,1].plot(r_kpc, u_mag, 'magenta', lw=2, label='Mag')
axs[0,1].plot(r_kpc, u_therm, 'white', lw=2, label='Therm')
axs[0,1].plot(r_kpc, u_cr, 'purple', lw=2, label='CR')
axs[0,1].set_title('Radial Energy Densities')
axs[0,1].set_xlabel('Radius [kpc]')
axs[0,1].set_ylabel('Energy Density [J/m³]')
axs[0,1].set_yscale('log')
axs[0,1].legend()
axs[0,1].grid(alpha=0.3)

# 3. Energy Ratios (key diagnostic)
axs[1,0].plot(r_kpc, u_mag / (u_rot_dm + 1e-30), 'magenta', lw=2, label='Mag / Rot(DM)')
axs[1,0].plot(r_kpc, u_turb / (u_rot_dm + 1e-30), 'lime', lw=2, label='Turb / Rot(DM)')
axs[1,0].plot(r_kpc, u_cr / (u_rot_dm + 1e-30), 'purple', lw=2, label='CR / Rot(DM)')
axs[1,0].plot(r_kpc, u_rot_plasma / (u_rot_dm + 1e-30), 'cyan', lw=2, label='Plasma Rot / DM Rot')
axs[1,0].set_title('Energy Density Ratios vs DM')
axs[1,0].set_xlabel('Radius [kpc]')
axs[1,0].set_ylabel('Ratio')
axs[1,0].legend()
axs[1,0].grid(alpha=0.3)

# 4. Bar Chart (average)
labels = ['Rot Plasma', 'Rot DM', 'Turb', 'Mag', 'Therm', 'CR']
values = [np.mean(u_rot_plasma), np.mean(u_rot_dm), np.mean(u_turb), np.mean(u_mag), np.mean(u_therm), np.mean(u_cr)]
axs[1,1].bar(labels, values, color=['cyan', 'orange', 'lime', 'magenta', 'white', 'purple'])
axs[1,1].set_title('Average Energy Densities')
axs[1,1].set_ylabel('Energy Density [J/m³]')
axs[1,1].tick_params(axis='x', rotation=45)

plt.suptitle('v0.4.1 Galactic Rotation + Radial Energy Ratio Analysis')
plt.tight_layout()
plt.show()
