import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 v1.8 — Heavily Stabilized 3D MHD (Self-Critique Fixes)")

N = 72
L = 45.0
dx = L / N
x = y = z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

mu0 = 4 * np.pi * 1e-7
eta = 0.008          # high resistivity
nu = 0.015           # artificial viscosity

r_cyl = np.sqrt(X**2 + Y**2)
r_sph = np.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

rho = 3.5e-24 * np.exp(-r_cyl / 12.0) * np.exp(-np.abs(Z)/4.5)
J_z = 6.0e17 * np.exp(-r_cyl**2 / (2*9.0**2))

v0 = 230 * 1000.0
v_theta = v0 * (1 - np.exp(-r_cyl / 8.0))
vx = -v_theta * (Y / (r_cyl + 1e-8))
vy =  v_theta * (X / (r_cyl + 1e-8))
vz = np.zeros_like(X, dtype=np.float32)

Bx = By = Bz = np.zeros_like(X, dtype=np.float32) * 5e-10
p = 1.2e-12 * rho

M_vir = 1.2e12 * 1.989e30
rs = 20.0 * 3.086e19
def nfw_mass(r):
    x = r / rs
    m = np.log(1 + x) - x / (1 + x)
    return M_vir * (m / (np.log(2) - 0.5))
M_enc_dm = nfw_mass(r_sph)

dt = 0.00008
steps = 800
max_v = 280 * 1000.0

time_steps, e_mag_list, e_kin_list, e_therm_list = [], [], [], []

print("Starting heavily stabilized run...\n")

start = time.time()

for step in range(steps):
    # CFL
    v_tot = np.sqrt(vx**2 + vy**2 + vz**2)
    cs = np.sqrt(1.4 * p / (rho + 1e-30))
    dt = min(0.00015, 0.35 * dx / (v_tot.max() + cs.max() + 1e-8))

    # Induction + resistivity
    vB_x = vy*Bz - vz*By
    vB_y = vz*Bx - vx*Bz
    vB_z = vx*By - vy*Bx

    dBx_dt = (np.gradient(vB_z, dx, axis=1) - np.gradient(vB_y, dx, axis=2)) + eta * np.gradient(np.gradient(Bx, dx, axis=0), dx, axis=0)
    dBy_dt = (np.gradient(vB_x, dx, axis=2) - np.gradient(vB_z, dx, axis=0)) + eta * np.gradient(np.gradient(By, dx, axis=1), dx, axis=1)
    dBz_dt = (np.gradient(vB_y, dx, axis=0) - np.gradient(vB_x, dx, axis=1)) + eta * np.gradient(np.gradient(Bz, dx, axis=2), dx, axis=2)

    Bx += dt * dBx_dt
    By += dt * dBy_dt
    Bz += dt * dBz_dt

    # Stronger cleaning
    divB = (np.gradient(Bx, dx, axis=0) + np.gradient(By, dx, axis=1) + np.gradient(Bz, dx, axis=2))
    Bx -= 0.4 * np.gradient(divB, dx, axis=0)
    By -= 0.4 * np.gradient(divB, dx, axis=1)
    Bz -= 0.4 * np.gradient(divB, dx, axis=2)

    # Forces
    Jx = (np.gradient(Bz, dx, axis=1) - np.gradient(By, dx, axis=2)) / mu0
    Jy = (np.gradient(Bx, dx, axis=2) - np.gradient(Bz, dx, axis=0)) / mu0
    Jz_total = J_z + (np.gradient(By, dx, axis=0) - np.gradient(Bx, dx, axis=1)) / mu0

    Fx = Jy * Bz - Jz_total * By
    Fy = Jz_total * Bx - Jx * Bz
    Fz = Jx * By - Jy * Bx

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

    # Artificial viscosity
    divv = np.gradient(vx, dx, axis=0) + np.gradient(vy, dx, axis=1) + np.gradient(vz, dx, axis=2)
    Fx -= nu * divv * vx
    Fy -= nu * divv * vy
    Fz -= nu * divv * vz

    # Update with limiter
    vx += dt * Fx / (rho + 1e-30)
    vy += dt * Fy / (rho + 1e-30)
    vz += dt * Fz / (rho + 1e-30)

    v_tot = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = v_tot > max_v
    if np.any(mask):
        factor = max_v / (v_tot[mask] + 1e-30)
        vx[mask] *= factor
        vy[mask] *= factor
        vz[mask] *= factor

    # Continuity
    div_v = (np.gradient(rho*vx, dx, axis=0) + np.gradient(rho*vy, dx, axis=1) + np.gradient(rho*vz, dx, axis=2))
    rho += dt * (-div_v)
    rho = np.maximum(rho, 5e-28)

    if step % 100 == 0 or step == steps-1:
        vol = dx**3
        e_mag = np.sum((Bx**2 + By**2 + Bz**2) / (2*mu0)) * vol
        e_kin = np.sum(0.5 * rho * v_tot**2) * vol
        e_therm = np.sum(1.5 * p) * vol

        time_steps.append(step*dt)
        e_mag_list.append(e_mag)
        e_kin_list.append(e_kin)
        e_therm_list.append(e_therm)

        Bmax = np.sqrt(Bx**2 + By**2 + Bz**2).max() * 1e6
        vmax = v_tot.max() / 1000
        beta = np.mean(p / ((Bx**2 + By**2 + Bz**2)/(2*mu0) + 1e-30))
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s | β = {beta:.3f}")
      print(f"\nFinished in {time.time() - start:.1f} s")

# ====================== PLOTS ======================
mid = N // 2

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(time_steps, e_mag_list, 'magenta', lw=2, label='Magnetic')
plt.plot(time_steps, e_kin_list, 'cyan', lw=2, label='Kinetic')
plt.plot(time_steps, e_therm_list, 'lime', lw=2, label='Thermal')
plt.title('Energy Evolution')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
total = np.array(e_mag_list) + np.array(e_kin_list) + np.array(e_therm_list)
plt.plot(time_steps, np.array(e_mag_list)/total, 'magenta', label='Mag fraction')
plt.plot(time_steps, np.array(e_kin_list)/total, 'cyan', label='Kin fraction')
plt.title('Energy Fractions')
plt.xlabel('Time (s)')
plt.ylabel('Fraction')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.imshow(np.sqrt(Bx[:,:,mid]**2 + By[:,:,mid]**2), cmap='magma')
plt.title('|B| mid-plane')
plt.colorbar()

plt.subplot(2, 2, 4)
r_plot = r_cyl[:,:,mid].flatten()
v_phi = (X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / (r_plot + 1e-8)
v_phi = v_phi.flatten() / 1000
bins = np.linspace(0, L/2, 50)
v_rot = [np.mean(np.abs(v_phi[(r_plot > bins[i]) & (r_plot < bins[i+1])])) 
         if np.any((r_plot > bins[i]) & (r_plot < bins[i+1])) else 0 for i in range(len(bins)-1)]
plt.plot(bins[:-1], v_rot, 'cyan', lw=2.5)
plt.axhline(230, color='red', ls='--')
plt.title('Rotation Curve')
plt.xlabel('Radius')
plt.ylabel('Velocity [km/s]')
plt.grid(True)

plt.suptitle('v1.6.7 Stable 3D MHD with Artificial Viscosity')
plt.tight_layout()
plt.show()

