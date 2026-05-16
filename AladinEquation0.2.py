import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ====================== CONSTANTS ======================
mu0 = 4 * np.pi * 1e-7
gamma = 5.0 / 3.0
G = 6.67430e-11
kpc_to_m = 3.086e19

N = 300
r_max_kpc = 25.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * kpc_to_m
dr = r[1] - r[0]
vol = 2 * np.pi * r * dr

# ====================== TUNED PARAMETERS ======================
I_total = 5e21                  # ← Increased significantly (try 1e21 to 2e22)
M_bary_total = 8e10 * 1.989e30
alpha_A = 0.12                  # Try 0.0 vs 0.12
sigma_kpc = 4.5
sigma = sigma_kpc * kpc_to_m

J_z = (I_total / (2 * np.pi * sigma**2)) * np.exp(-(r / sigma)**2)

rho0 = 2e-20 * np.exp(-r_kpc / 4.0)

def initial_v_theta(r_kpc):
    v_inner = 180 * np.sqrt(r_kpc / 2.0)
    return np.where(r_kpc < 6, v_inner, 230) * 1000

v_theta0 = initial_v_theta(r_kpc)

M_enc_bary = M_bary_total * (1 - np.exp(-r / (3*sigma)))
def gravity_acc(r):
    return G * M_enc_bary / (r**2 + 1e-12)

# ====================== RHS (same as before) ======================
def rhs(t, y, alpha_A):
    rho = y[0:N]
    vr = y[N:2*N]
    v_theta = y[2*N:3*N]
    p = y[3*N:4*N]
    
    I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
    B = mu0 * I_enc / (2 * np.pi * r + 1e-20)
    
    dB_dr = np.gradient(B, r)
    JxB_r = - (B / mu0) * (dB_dr + B / (r + 1e-20))
    
    aladin_factor = 1.0 + alpha_A * np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30)
    
    dp_dr = np.gradient(p, r)
    g_r = gravity_acc(r)
    
    a_r_total = JxB_r * aladin_factor - dp_dr / (rho + 1e-30) - g_r
    
    dvr_dt = a_r_total - (v_theta**2 / r)
    dvtheta_dt = - (vr * v_theta / r)
    
    div_v = (1/r) * np.gradient(r * vr, r)
    drho_dt = -rho * div_v - vr * np.gradient(rho, r)
    dp_dt = -gamma * p * div_v
    
    return np.concatenate((drho_dt, dvr_dt, dvtheta_dt, dp_dt))

# Run
y0 = np.concatenate((rho0, np.zeros_like(r), v_theta0, rho0 * 1e10))
sol = solve_ivp(lambda t,y: rhs(t, y, alpha_A), [0, 8.0], y0, 
                method='LSODA', rtol=1e-5, atol=1e-8)

# ====================== RESULTS ======================
final = -1
v_theta_final = sol.y[2*N:3*N, final] / 1000

I_enc_f = np.cumsum(2 * np.pi * r * J_z * dr)
B_f = mu0 * I_enc_f / (2 * np.pi * r)

print(f"Peak B field: {B_f.max()*1e6:.2f} μG")
print(f"Peak rotation velocity: {v_theta_final.max():.1f} km/s")

# Energy check (same as before)
vol = 2 * np.pi * r * dr
E_kin = np.sum(0.5 * sol.y[0:N,final] * (sol.y[N:2*N,final]**2 + sol.y[2*N:3*N,final]**2) * vol)
E_therm = np.sum(sol.y[3*N:4*N,final] / (gamma-1) * vol)
E_mag = np.sum(B_f**2 / (2*mu0) * vol)
print(f"E_mag = {E_mag:.2e} J | E_kin = {E_kin:.2e} J")

# Plot (simplified)
plt.style.use('dark_background')
plt.figure(figsize=(12, 8))
plt.plot(r_kpc, v_theta_final, 'cyan', lw=3, label='Simulated v_θ')
plt.axhline(230, color='white', linestyle='--', label='Milky Way typical')
plt.title(f'Rotation Curve\nPeak B = {B_f.max()*1e6:.2f} μG | α_A = {alpha_A}')
plt.xlabel('Radius [kpc]')
plt.ylabel('Velocity [km/s]')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
