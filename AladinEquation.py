import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ====================== CONSTANTS & UNITS ======================
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
vol = 2 * np.pi * r * dr   # volume element (per unit z-length)

# ====================== PARAMETERS ======================
I_total = 3e19
M_bary_total = 8e10 * 1.989e30
alpha_A = 0.08                    # Change to 0.0 for pure MHD
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

# ====================== RHS ======================
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

# ====================== RUN WITH ENERGY TRACKING ======================
y0 = np.concatenate((rho0, np.zeros_like(r), v_theta0, rho0 * 1e10))

sol = solve_ivp(lambda t,y: rhs(t, y, alpha_A), [0, 8.0], y0, 
                method='LSODA', rtol=1e-5, atol=1e-8, dense_output=True)

# Compute energies at saved time steps
times = sol.t
E_kin = []
E_therm = []
E_mag = []
E_total = []

for i in range(len(times)):
    y = sol.y[:, i]
    rho = y[0:N]
    vr = y[N:2*N]
    v_theta = y[2*N:3*N]
    p = y[3*N:4*N]
    
    I_enc = np.cumsum(2 * np.pi * r * J_z * dr)
    B = mu0 * I_enc / (2 * np.pi * r + 1e-20)
    
    e_kin = 0.5 * rho * (vr**2 + v_theta**2)
    e_therm = p / (gamma - 1)
    e_mag = B**2 / (2 * mu0)
    
    E_kin.append(np.sum(e_kin * vol))
    E_therm.append(np.sum(e_therm * vol))
    E_mag.append(np.sum(e_mag * vol))
    E_total.append(E_kin[-1] + E_therm[-1] + E_mag[-1])

E_kin = np.array(E_kin)
E_therm = np.array(E_therm)
E_mag = np.array(E_mag)
E_total = np.array(E_total)

drift = (E_total[-1] - E_total[0]) / E_total[0] * 100

print(f"\n=== ENERGY CONSERVATION CHECK ===")
print(f"Initial Total Energy : {E_total[0]:.2e} J")
print(f"Final Total Energy   : {E_total[-1]:.2e} J")
print(f"Energy Drift         : {drift:.3f}%")
print(f"Peak B field         : {np.max(B)*1e6:.2f} μG")

# ====================== PLOTS ======================
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Rotation Curve
v_theta_final = sol.y[2*N:3*N, -1] / 1000
axs[0,0].plot(r_kpc, v_theta_final, 'cyan', lw=3, label='v_θ (final)')
axs[0,0].set_title('Final Rotation Curve')
axs[0,0].set_xlabel('Radius [kpc]')
axs[0,0].set_ylabel('Velocity [km/s]')
axs[0,0].legend()
axs[0,0].grid(alpha=0.3)

# Energy Evolution
axs[0,1].plot(times, E_kin/1e40, label='Kinetic', lw=2)
axs[0,1].plot(times, E_therm/1e40, label='Thermal', lw=2)
axs[0,1].plot(times, E_mag/1e40, label='Magnetic', lw=2)
axs[0,1].plot(times, E_total/1e40, 'white', lw=2.5, label='Total')
axs[0,1].set_title('Energy Evolution')
axs[0,1].set_xlabel('Time')
axs[0,1].set_ylabel('Energy (×10⁴⁰ J)')
axs[0,1].legend()

# Density & B field
axs[1,0].plot(r_kpc, sol.y[0:N, -1], 'white', lw=2)
axs[1,0].set_title('Final Density')

axs[1,1].plot(r_kpc, B*1e6, 'magenta', lw=2)
axs[1,1].set_title('B_θ [μG]')

plt.suptitle(f'Z-Pinch Galactic Simulation with Energy Conservation\nα_A = {alpha_A} | Energy Drift = {drift:.3f}%', fontsize=16)
plt.tight_layout()
plt.show()
