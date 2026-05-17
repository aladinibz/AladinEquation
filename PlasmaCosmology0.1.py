import numpy as np
import matplotlib.pyplot as plt

print("🌌 v0.3.0 Galactic Rotation Curves — Dynamic Plasma Model")

# ====================== SETUP ======================
N = 300
r_max_kpc = 30.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * 3.086e19
dr = r[1] - r[0]

mu0 = 4 * np.pi * 1e-7
G = 6.6743e-11
gamma = 5.0/3.0

h_kpc = 0.4
h = h_kpc * 3.086e19

# ====================== PARAMETERS ======================
I_total = 6e21
M_bary_total = 6e10 * 1.989e30
alpha_A = 0.15
sigma_kpc = 5.5
sigma = sigma_kpc * 3.086e19

J_z = (I_total / (2 * np.pi * sigma**2)) * np.exp(-r**2 / (2*sigma**2))

rho = 1.5e-20 * np.exp(-r_kpc / 6.0)   # more realistic average density

# Initial rotation
v_theta = np.zeros_like(r)
v_theta = (180 * np.sqrt(np.maximum(r_kpc/4, 0.5))) * 1000
v_theta[r_kpc > 8] = 230 * 1000

p = 1e8 * rho

dt = 0.0015
steps = 2000

print("Running dynamic evolution...")

for step in range(steps):
    I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
    B = mu0 * I_enc / (2 * np.pi * r + 1e-20)
    dB_dr = np.gradient(B, r)
    JxB_r = - (B / mu0) * (dB_dr + B / (r + 1e-20))
    
    dp_dr = np.gradient(p, r)
    g_r = G * M_bary_total * (1 - np.exp(-r / (3*sigma))) / (r**2 + 1e10)
    
    # Turb + CR pressure
    sigma_turb = 25*1000
    P_turb = rho * sigma_turb**2
    a_turb = -np.gradient(P_turb, r) / (rho + 1e-30)
    
    P_cr = 5e-13 * np.exp(-r_kpc/8)
    a_cr = -np.gradient(P_cr, r) / (rho + 1e-30)
    
    boost = np.clip(np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30), 0, 8)
    aladin_factor = 1.0 + alpha_A * boost
    
    a_r_total = JxB_r * aladin_factor - dp_dr / (rho + 1e-30) - g_r + a_turb + a_cr
    
    # Update rotation (force balance + angular momentum)
    v_theta += dt * (a_r_total * 0.0 - v_theta**2 / r)   # centrifugal
    # Relax toward equilibrium
    v_eq = np.sqrt(np.maximum(r * (-a_r_total), 0))
    v_theta = 0.98 * v_theta + 0.02 * v_eq
    
    # Simple continuity (no strong radial flow)
    rho *= 0.999   # mild adjustment

    if step % 400 == 0:
        print(f"Step {step:4d} | Peak v = {v_theta.max()/1000:.1f} km/s | Max B = {B.max()*1e6:.2f} μG")

# Final energies
vol = 2 * np.pi * r * dr * h
E_rot = np.sum(0.5 * rho * v_theta**2 * vol)
E_mag = np.sum(B**2 / (2*mu0) * vol)
print(f"\nFinal Peak v = {v_theta.max()/1000:.1f} km/s")
print(f"Rotational KE ~ {E_rot:.2e} J")
print(f"Magnetic KE   ~ {E_mag:.2e} J")

plt.style.use('dark_background')
plt.figure(figsize=(12, 7))
plt.plot(r_kpc, v_theta/1000, 'cyan', lw=3, label='Plasma Z-Pinch Model')
plt.axhline(230, color='yellow', ls='--', label='Milky Way ~230 km/s')
plt.title('v0.3.0 Dynamic Plasma Rotation Curve')
plt.xlabel('Radius [kpc]')
plt.ylabel('Velocity [km/s]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
