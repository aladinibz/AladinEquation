import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys

# ====================== CONSTANTS ======================
mu0 = 4 * np.pi * 1e-7
gamma = 5.0 / 3.0
G = 6.67430e-11
kpc_to_m = 3.086e19

# ====================== GRID ======================
N = 300
r_max_kpc = 25.0
r_kpc = np.linspace(0.1, r_max_kpc, N)
r = r_kpc * kpc_to_m
dr = r[1] - r[0]

# ====================== PARAMETERS ======================
I_total = 5e21
M_bary_total = 8e10 * 1.989e30
alpha_A = 0.12
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

# ====================== SAFE RHS ======================
def rhs(t, y, alpha):
    try:
        rho = y[0:N]
        vr = y[N:2*N]
        v_theta = y[2*N:3*N]
        p = y[3*N:4*N]
        
        # Safety checks
        rho = np.maximum(rho, 1e-30)
        r_safe = np.maximum(r, 1e-10)
        
        I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
        B = mu0 * I_enc / (2 * np.pi * r_safe)
        
        dB_dr = np.gradient(B, r)
        JxB_r = - (B / mu0) * (dB_dr + B / r_safe)
        
        aladin_factor = 1.0 + alpha * np.abs(J_z * B) / (np.abs(JxB_r) + 1e-30)
        
        dp_dr = np.gradient(p, r)
        g_r = gravity_acc(r)
        
        a_r_total = JxB_r * aladin_factor - dp_dr / rho - g_r
        
        dvr_dt = a_r_total - (v_theta**2 / r_safe)
        dvtheta_dt = - (vr * v_theta / r_safe)
        
        div_v = (1/r_safe) * np.gradient(r * vr, r)
        drho_dt = -rho * div_v - vr * np.gradient(rho, r)
        dp_dt = -gamma * p * div_v
        
        return np.concatenate((drho_dt, dvr_dt, dvtheta_dt, dp_dt))
    
    except Exception as e:
        print(f"Error in RHS at t={t}: {e}")
        return np.zeros_like(y)

# ====================== RUN WITH ERROR HANDLING ======================
def run_case(alpha, label):
    try:
        y0 = np.concatenate((rho0.copy(), np.zeros_like(r), v_theta0.copy(), rho0 * 1e10))
        print(f"Running {label}...")
        
        sol = solve_ivp(lambda t,y: rhs(t, y, alpha), [0, 8.0], y0,
                        method='LSODA', rtol=1e-5, atol=1e-8, max_step=0.1)
        
        if not sol.success:
            print(f"Warning: {label} solver did not succeed. Message: {sol.message}")
        
        v_final = sol.y[2*N:3*N, -1] / 1000
        print(f"Peak velocity ({label}): {v_final.max():.1f} km/s")
        return sol, v_final
        
    except Exception as e:
        print(f"CRITICAL ERROR in {label}: {e}")
        sys.exit(1)

# ====================== MAIN ======================
print("=== Aladin Equation Test ===\n")
sol_pure, v_pure = run_case(0.0, "Pure MHD")
sol_aladin, v_aladin = run_case(alpha_A, f"With Aladin α={alpha_A}")

# Final diagnostics
I_enc_f = np.cumsum(2 * np.pi * r * J_z * dr)
B_f = mu0 * I_enc_f / (2 * np.pi * np.maximum(r, 1e-10))

print(f"Peak B field: {B_f.max()*1e6:.2f} μG")

# ====================== PLOTS ======================
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

axs[0,0].plot(r_kpc, v_pure, 'lime', lw=2.5, label='Pure MHD')
axs[0,0].plot(r_kpc, v_aladin, 'cyan', lw=3.5, label=f'Aladin (α={alpha_A})')
axs[0,0].axhline(230, color='white', ls='--', label='Milky Way ~230 km/s')
axs[0,0].set_title('Rotation Curve Comparison')
axs[0,0].set_xlabel('Radius [kpc]')
axs[0,0].set_ylabel('Velocity [km/s]')
axs[0,0].legend()
axs[0,0].grid(alpha=0.3)

axs[0,1].plot(r_kpc, sol_aladin.y[0:N, -1], 'white', lw=2)
axs[0,1].set_title('Final Density Profile')

axs[1,0].plot(r_kpc, B_f*1e6, 'magenta', lw=2)
axs[1,0].set_title('B_θ Field [μG]')

# Aladin factor
JxB_mag = np.abs(- (B_f / mu0) * (np.gradient(B_f, r) + B_f / np.maximum(r,1e-10)))
aladin_factor = 1.0 + alpha_A * np.abs(J_z * B_f) / (JxB_mag + 1e-30)
axs[1,1].plot(r_kpc, aladin_factor, 'orange', lw=2.5)
axs[1,1].set_title('Aladin Amplification Factor')

plt.suptitle('Aladin Equation — Robust Test with Error Handling', fontsize=16)
plt.tight_layout()
plt.show()
