import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 v1.5.3 — Grid Convergence Test (N=64, 96, 128)")

# ====================== PARAMETERS ======================
L = 45.0
mu0 = 4 * np.pi * 1e-7
eta = 0.0009
dt = 0.00085
steps = 400          # Reduced for convergence testing speed

M_vir = 1.2e12 * 1.989e30
rs = 20.0 * 3.086e19

def run_simulation(N):
    dx = L / N
    x = y = z = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    r_cyl = np.sqrt(X**2 + Y**2)
    r_sph = np.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

    # Fields
    rho = 2.2e-24 * np.exp(-r_cyl / 13.0) * np.exp(-np.abs(Z)/5.0)
    J_z = 1.8e18 * np.exp(-r_cyl**2 / (2*5.5**2))
    
    v0 = 230 * 1000.0
    v_theta = v0 * (1 - np.exp(-r_cyl / 6.0))
    vx = -v_theta * (Y / (r_cyl + 1e-8))
    vy =  v_theta * (X / (r_cyl + 1e-8))
    vz = np.zeros_like(X, dtype=np.float32)

    Bx = By = Bz = np.zeros_like(X, dtype=np.float32) * 2e-9
    p = 4e-13 * rho

    # NFW
    def nfw_mass(r):
        x = r / rs
        m = np.log(1 + x) - x / (1 + x)
        return M_vir * (m / (np.log(2) - 0.5))
    M_enc_dm = nfw_mass(r_sph)

    start = time.time()
    for step in range(steps):
        # Induction
        vB_x = vy*Bz - vz*By
        vB_y = vz*Bx - vx*Bz
        vB_z = vx*By - vy*Bx
        
        dBx_dt = (np.gradient(vB_z, dx, axis=1) - np.gradient(vB_y, dx, axis=2)) + \
                 eta * np.gradient(np.gradient(Bx, dx, axis=0), dx, axis=0)
        dBy_dt = (np.gradient(vB_x, dx, axis=2) - np.gradient(vB_z, dx, axis=0)) + \
                 eta * np.gradient(np.gradient(By, dx, axis=1), dx, axis=1)
        dBz_dt = (np.gradient(vB_y, dx, axis=0) - np.gradient(vB_x, dx, axis=1)) + \
                 eta * np.gradient(np.gradient(Bz, dx, axis=2), dx, axis=2)
        
        Bx += dt * dBx_dt
        By += dt * dBy_dt
        Bz += dt * dBz_dt
        
        # Divergence cleaning
        divB = (np.gradient(Bx, dx, axis=0) + np.gradient(By, dx, axis=1) + np.gradient(Bz, dx, axis=2))
        Bx -= 0.28 * np.gradient(divB, dx, axis=0)
        By -= 0.28 * np.gradient(divB, dx, axis=1)
        Bz -= 0.28 * np.gradient(divB, dx, axis=2)
        
        # Currents & Lorentz
        Jx = (np.gradient(Bz, dx, axis=1) - np.gradient(By, dx, axis=2)) / mu0
        Jy = (np.gradient(Bx, dx, axis=2) - np.gradient(Bz, dx, axis=0)) / mu0
        Jz_total = J_z + (np.gradient(By, dx, axis=0) - np.gradient(Bx, dx, axis=1)) / mu0
        
        Fx = Jy * Bz - Jz_total * By
        Fy = Jz_total * Bx - Jx * Bz
        Fz = Jx * By - Jy * Bx
        
        # Gravity + Pressure
        grav_dm = -6.6743e-11 * M_enc_dm / (r_sph**2 + 1e-8)
        Fx += grav_dm * (X / r_sph)
        Fy += grav_dm * (Y / r_sph)
        Fz += grav_dm * (Z / r_sph)
        
        dpdx = np.gradient(p, dx, axis=0)
        dpdy = np.gradient(p, dx, axis=1)
        dpdz = np.gradient(p, dx, axis=2)
        Fx -= dpdx
        Fy -= dpdy
        Fz -= dpdz
        
        vx += dt * Fx / (rho + 1e-30)
        vy += dt * Fy / (rho + 1e-30)
        vz += dt * Fz / (rho + 1e-30)
        
        # Continuity
        div_v = (np.gradient(rho*vx, dx, axis=0) + 
                 np.gradient(rho*vy, dx, axis=1) + 
                 np.gradient(rho*vz, dx, axis=2))
        rho += dt * (-div_v)
        rho = np.maximum(rho, 5e-28)
    
    runtime = time.time() - start
    Bmax = np.sqrt(Bx**2 + By**2 + Bz**2).max() * 1e6
    vmax = np.sqrt(vx**2 + vy**2 + vz**2).max() / 1000
    
    # Mid-plane rotation curve
    mid = N // 2
    r_plot = r_cyl[:,:,mid].flatten()
    v_phi = (X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / (r_plot + 1e-8)
    v_phi = v_phi.flatten() / 1000
    
    return {
        'N': N,
        'Bmax_μG': Bmax,
        'vmax_kms': vmax,
        'runtime': runtime,
        'r_plot': r_plot,
        'v_phi': v_phi
    }

# ====================== RUN CONVERGENCE TEST ======================
resolutions = [64, 96, 112]
results = []

for N in resolutions:
    print(f"\n=== Running N = {N} ===")
    res = run_simulation(N)
    results.append(res)
    print(f"N={N:3d} | Bmax={res['Bmax_μG']:.2f} μG | vmax={res['vmax_kms']:.1f} km/s | Time={res['runtime']:.1f}s")

# ====================== PLOTS ======================
plt.figure(figsize=(15, 8))

# Convergence of key quantities
Ns = [r['N'] for r in results]
Bmaxs = [r['Bmax_μG'] for r in results]
vmaxs = [r['vmax_kms'] for r in results]

plt.subplot(2, 2, 1)
plt.plot(Ns, Bmaxs, 'o-', color='magenta')
plt.title('Max |B| vs Grid Resolution')
plt.xlabel('N')
plt.ylabel('Max |B| [μG]')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(Ns, vmaxs, 'o-', color='cyan')
plt.title('Max Velocity vs Grid Resolution')
plt.xlabel('N')
plt.ylabel('Max |v| [km/s]')
plt.grid(True)

# Rotation curves comparison
plt.subplot(2, 1, 2)
for res in results:
    r_plot = res['r_plot']
    v_phi = res['v_phi']
    bins = np.linspace(0, L/2, 40)
    v_rot = [np.mean(np.abs(v_phi[(r_plot > bins[i]) & (r_plot < bins[i+1])])) 
             if np.any((r_plot > bins[i]) & (r_plot < bins[i+1])) else 0 
             for i in range(len(bins)-1)]
    plt.plot(bins[:-1], v_rot, lw=2, label=f'N={res["N"]}')

plt.axhline(230, color='red', ls='--', label='Observed ~230 km/s')
plt.title('Rotation Curve Convergence')
plt.xlabel('Radius')
plt.ylabel('Velocity [km/s]')
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle('Grid Convergence Test — v1.5.3 Pure First-Principles 3D MHD + NFW')
plt.tight_layout()
plt.show()
