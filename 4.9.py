import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 Plasma Cosmology v4.9 — Improved Pure Plasma Galactic Mode")

# ====================== CONTROLS ======================
RUN_COMPARISON = True     # Run both pure plasma and with DM
USE_CR = True

# ====================== GRID ======================
N = 80
L = 60.0
dx = L / N
x = y = z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

r_cyl = np.sqrt(X**2 + Y**2)
r_sph = np.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

mu0 = 4 * np.pi * 1e-7
gamma = 5.0/3.0
CFL = 0.35
ch = 2.0
kappa = 0.5
steps = 1200
max_v = 350 * 1000.0
vol = dx**3

def run_galaxy(use_dm):
    # Reset fields
    rho = 1.8e-21 * np.exp(-r_cyl / 12.0) * np.exp(-np.abs(Z)/4.0)
    E_total = np.zeros((N, N, N), dtype=np.float32)
    vx = np.zeros((N, N, N), dtype=np.float32)
    vy = np.zeros((N, N, N), dtype=np.float32)
    vz = np.zeros((N, N, N), dtype=np.float32)
    u_cr = 1.6e-13 * np.exp(-r_cyl / 15.0) * np.exp(-np.abs(Z)/6.0) if USE_CR else np.zeros_like(rho)

    # Much weaker imposed current (more self-consistent)
    J_z = 2.5e17 * np.exp(-r_cyl**2 / (2*12.0**2))

    Bx = np.zeros((N+1, N, N), dtype=np.float32)
    By = np.zeros((N, N+1, N), dtype=np.float32)
    Bz = np.zeros((N, N, N+1), dtype=np.float32)
    psi = np.zeros((N, N, N), dtype=np.float32)

    # Weak seed B
    for k in range(N+1):
        zf = -L/2 + k*dx
        r2d = np.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
        Bz[:,:,k] = 1.8e-10 * np.exp(-r2d**2 / 250.0) * np.exp(-zf**2 / 40.0)

    M_enc_dm = nfw_mass(r_sph) if use_dm else np.zeros_like(r_sph)

    # Initial energy
    p_th = 2e-12 * rho
    kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_total = p_th / (gamma - 1) + kin + u_cr

    e_kin_h = []
    e_mag_h = []
    e_therm_h = []
    e_cr_h = []
    Lz_h = []   # angular momentum

    for step in range(steps):
        Bx_c = (Bx[:-1] + Bx[1:]) / 2
        By_c = (By[:,:-1] + By[:,1:]) / 2
        Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2
        B2 = Bx_c**2 + By_c**2 + Bz_c**2

        # CFL + CT + Hyperbolic cleaning (stable block)
        # [CT + cleaning from previous versions - kept for brevity]

        # Lorentz
        Jx = (np.gradient(Bz_c, dx, axis=1) - np.gradient(By_c, dx, axis=2)) / mu0
        Jy = (np.gradient(Bx_c, dx, axis=2) - np.gradient(Bz_c, dx, axis=0)) / mu0
        Jz_total = J_z + (np.gradient(By_c, dx, axis=0) - np.gradient(Bx_c, dx, axis=1)) / mu0

        Fx = Jy * Bz_c - Jz_total * By_c
        Fy = Jz_total * Bx_c - Jx * Bz_c
        Fz = Jx * By_c - Jy * Bx_c

        if use_dm:
            grav = -6.6743e-11 * M_enc_dm / (r_sph**2 + 1e-8)
            Fx += grav * (X / r_sph)
            Fy += grav * (Y / r_sph)
            Fz += grav * (Z / r_sph)

        P_cr = u_cr / 3.0 if USE_CR else 0.0
        P_tot = p_th + P_cr + B2 / (2*mu0)

        Fx -= np.gradient(P_tot, dx, axis=0)
        Fy -= np.gradient(P_tot, dx, axis=1)
        Fz -= np.gradient(P_tot, dx, axis=2)

        # Update
        vx += dt * Fx / (rho + 1e-30)
        vy += dt * Fy / (rho + 1e-30)
        vz += dt * Fz / (rho + 1e-30)

        v_tot = np.sqrt(vx**2 + vy**2 + vz**2)
        vx = np.clip(vx, -max_v, max_v)
        vy = np.clip(vy, -max_v, max_v)
        vz = np.clip(vz, -max_v, max_v)

        div_v = (np.gradient(rho*vx, dx, axis=0) + np.gradient(rho*vy, dx, axis=1) + np.gradient(rho*vz, dx, axis=2))
        rho += dt * (-div_v)
        rho = np.maximum(rho, 1e-25)

        # Simple CR transport
        if USE_CR:
            div_cr = (np.gradient(u_cr*vx, dx, axis=0) + np.gradient(u_cr*vy, dx, axis=1) + np.gradient(u_cr*vz, dx, axis=2))
            lap_cr = sum(np.gradient(np.gradient(u_cr, dx, axis=i), dx, axis=i) for i in range(3))
            source = 2.5e-15 * np.exp(-r_cyl / 8.0) * np.exp(-np.abs(Z)/3.0)
            u_cr += dt * (-div_cr + 3e-4 * lap_cr + source)

        # Energy & Angular Momentum diagnostics
        if step % 100 == 0 or step == steps-1:
            kin = np.sum(0.5 * rho * v_tot**2) * vol
            mag = np.sum(B2 / (2*mu0)) * vol
            therm = np.sum(p_th / (gamma-1)) * vol
            cr_e = np.sum(u_cr) * vol if USE_CR else 0
            total = kin + mag + therm + cr_e

            Lz = np.sum(rho * (X*vy - Y*vx) * r_cyl) * vol   # approximate angular momentum

            e_kin_h.append(kin)
            e_mag_h.append(mag)
            e_therm_h.append(therm)
            e_cr_h.append(cr_e)
            Lz_h.append(Lz)

            Bmax = np.sqrt(B2).max() * 1e6
            vmax = v_tot.max() / 1000
            print(f"Step {step:4d} | B = {Bmax:.2f} μG | v = {vmax:.1f} km/s | Lz = {Lz:.2e}")

    return e_kin_h, e_mag_h, e_therm_h, e_cr_h, Lz_h, v_tot, Bx_c, By_c, Bz_c, r_cyl

# Run comparison
if RUN_COMPARISON:
    print("Running Pure Plasma...")
    kin_p, mag_p, therm_p, cr_p, Lz_p, v_p, Bx_p, By_p, Bz_p, r_p = run_galaxy(False)
    print("\nRunning with DM...")
    kin_d, mag_d, therm_d, cr_d, Lz_d, v_d, Bx_d, By_d, Bz_d, r_d = run_galaxy(True)
else:
    kin_p, mag_p, therm_p, cr_p, Lz_p, v_p, Bx_p, By_p, Bz_p, r_p = run_galaxy(USE_DM)

# ====================== PLOTS ======================
mid = N // 2
r_plot = r_p[:,:,mid].flatten()
v_phi = (X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / (r_plot + 1e-8)
v_phi = v_phi.flatten() / 1000

bins = np.linspace(0, L/2, 60)
v_rot = [np.mean(np.abs(v_phi[(r_plot > bins[i]) & (r_plot < bins[i+1])])) 
         if np.any((r_plot > bins[i]) & (r_plot < bins[i+1])) else 0 for i in range(len(bins)-1)]

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(bins[:-1], v_rot, 'cyan', lw=2.5, label='Pure Plasma')
plt.axhline(230, color='red', ls='--', label='Observed ~230 km/s')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Rotation Curve')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(kin_p, label='Kinetic', color='cyan')
plt.plot(mag_p, label='Magnetic', color='magenta')
plt.plot(therm_p, label='Thermal', color='lime')
if USE_CR:
    plt.plot(cr_p, label='CR', color='orange')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.title('Energy Budget — Pure Plasma')
plt.legend()
plt.grid(True)

plt.suptitle('v4.9 Galactic Plasma Simulation — Pure Plasma Focus')
plt.tight_layout()
plt.show()
