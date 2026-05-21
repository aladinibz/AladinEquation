
import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 Plasma Cosmology v4.9 FIXED — Galactic Pure Plasma Mode")

# ====================== CONTROLS ======================
USE_DM = False          # Set True for comparison with DM halo
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
steps = 1000
max_v = 350 * 1000.0

# ====================== NFW MASS FUNCTION ======================
M_vir = 1.2e12 * 1.989e30   # Solar masses -> kg
rs = 22.0 * 3.086e19        # scale radius in meters
def nfw_mass(r):
    """NFW enclosed mass"""
    x = r / rs
    return M_vir * (np.log(1 + x) - x / (1 + x)) / (np.log(2) - 0.5)

M_enc_dm = nfw_mass(r_sph) if USE_DM else np.zeros_like(r_sph)

# ====================== INITIAL CONDITIONS ======================
rho = 1.8e-21 * np.exp(-r_cyl / 12.0) * np.exp(-np.abs(Z)/4.0)
vx = np.zeros((N, N, N), dtype=np.float32)
vy = np.zeros((N, N, N), dtype=np.float32)
vz = np.zeros((N, N, N), dtype=np.float32)

u_cr = 1.6e-13 * np.exp(-r_cyl / 15.0) * np.exp(-np.abs(Z)/6.0) if USE_CR else np.zeros_like(rho)

J_z = 2.8e17 * np.exp(-r_cyl**2 / (2*11.0**2))

Bx = np.zeros((N+1, N, N), dtype=np.float32)
By = np.zeros((N, N+1, N), dtype=np.float32)
Bz = np.zeros((N, N, N+1), dtype=np.float32)
psi = np.zeros((N, N, N), dtype=np.float32)

for k in range(N+1):
    zf = -L/2 + k*dx
    r2d = np.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
    Bz[:,:,k] = 2.2e-10 * np.exp(-r2d**2 / 220.0) * np.exp(-zf**2 / 35.0)

p_th = 2e-12 * rho
E_total = p_th / (gamma - 1) + 0.5*rho*(vx**2 + vy**2 + vz**2) + u_cr

print(f"Mode: {'DM + ' if USE_DM else ''}Pure Plasma {'+ CR' if USE_CR else ''}\n")
start = time.time()

for step in range(steps):
    Bx_c = (Bx[:-1] + Bx[1:]) / 2
    By_c = (By[:,:-1] + By[:,1:]) / 2
    Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2
    B2 = Bx_c**2 + By_c**2 + Bz_c**2

    # CFL
    vtot = np.sqrt(vx**2 + vy**2 + vz**2)
    p_thermal = (gamma-1) * (E_total - 0.5*rho*vtot**2 - B2/(2*mu0) - u_cr)
    cs = np.sqrt(gamma * np.maximum(p_thermal,0) / (rho + 1e-30))
    ca = np.sqrt(B2 / (mu0 * rho + 1e-30))
    cmax = vtot.max() + cs.max() + ca.max() + ch
    dt = CFL * dx / cmax

    # CT + EMFs + Hyperbolic Cleaning (stable)
    Ex = np.zeros((N, N+1, N+1), dtype=np.float32)
    Ey = np.zeros((N+1, N, N+1), dtype=np.float32)
    Ez = np.zeros((N+1, N+1, N), dtype=np.float32)

    Ex[:,1:,1:] = -(vy * Bz_c - vz * By_c)
    Ey[1:,:,1:] = -(vz * Bx_c - vx * Bz_c)
    Ez[1:,1:,:] = -(vx * By_c - vy * Bx_c)

    curlEx = ((Ez[1:-1,1:,:] - Ez[1:-1,:-1,:]) - (Ey[1:-1,:,1:] - Ey[1:-1,:,:-1])) / dx
    curlEy = ((Ex[:,1:-1,1:] - Ex[:,1:-1,:-1]) - (Ez[1:,1:-1,:] - Ez[:-1,1:-1,:])) / dx
    curlEz = ((Ey[1:,:,1:-1] - Ey[:-1,:,1:-1]) - (Ex[:,1:,1:-1] - Ex[:,:-1,1:-1])) / dx

    Bx[1:-1] += dt * curlEx
    By[:,1:-1] += dt * curlEy
    Bz[:,:,1:-1] += dt * curlEz

    # Hyperbolic cleaning
    divB = (np.gradient(Bx_c, dx, axis=0) + np.gradient(By_c, dx, axis=1) + np.gradient(Bz_c, dx, axis=2))
    psi -= dt * (ch**2 * divB + (ch / (kappa * dx)) * psi)
    Bx[1:-1] -= dt * np.gradient(psi, dx, axis=0)
    By[:,1:-1] -= dt * np.gradient(psi, dx, axis=1)
    Bz[:,:,1:-1] -= dt * np.gradient(psi, dx, axis=2)

    # Forces
    Jx = (np.gradient(Bz_c, dx, axis=1) - np.gradient(By_c, dx, axis=2)) / mu0
    Jy = (np.gradient(Bx_c, dx, axis=2) - np.gradient(Bz_c, dx, axis=0)) / mu0
    Jz_total = J_z + (np.gradient(By_c, dx, axis=0) - np.gradient(Bx_c, dx, axis=1)) / mu0

    Fx = Jy * Bz_c - Jz_total * By_c
    Fy = Jz_total * Bx_c - Jx * Bz_c
    Fz = Jx * By_c - Jy * Bx_c

    if USE_DM:
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

    if USE_CR:
        div_cr = (np.gradient(u_cr*vx, dx, axis=0) + np.gradient(u_cr*vy, dx, axis=1) + np.gradient(u_cr*vz, dx, axis=2))
        lap_cr = sum(np.gradient(np.gradient(u_cr, dx, axis=i), dx, axis=i) for i in range(3))
        source = 2.5e-15 * np.exp(-r_cyl / 8.0) * np.exp(-np.abs(Z)/3.0)
        u_cr += dt * (-div_cr + 3e-4 * lap_cr + source)

    if step % 100 == 0 or step == steps-1:
        Bmax = np.sqrt(B2).max() * 1e6
        vmax = v_tot.max() / 1000
        print(f"Step {step:4d} | Bmax = {Bmax:.2f} μG | vmax = {vmax:.1f} km/s")

print(f"Finished in {time.time() - start:.1f} s")

# ====================== ROTATION CURVE ======================
mid = N // 2
r_plot = r_cyl[:,:,mid].flatten()
v_phi = (X[:,:,mid]*vy[:,:,mid] - Y[:,:,mid]*vx[:,:,mid]) / (r_plot + 1e-8)
v_phi = v_phi.flatten() / 1000

bins = np.linspace(0, L/2, 60)
v_rot = [np.mean(np.abs(v_phi[(r_plot > bins[i]) & (r_plot < bins[i+1])])) 
         if np.any((r_plot > bins[i]) & (r_plot < bins[i+1])) else 0 for i in range(len(bins)-1)]

plt.figure(figsize=(12, 5))
mode = "Pure Plasma + CR" if not USE_DM else "Plasma + DM"
plt.plot(bins[:-1], v_rot, 'cyan', lw=2.5, label=mode)
plt.axhline(230, color='red', ls='--', label='Observed flat ~230 km/s')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Galactic Rotation Curve')
plt.legend()
plt.grid(True)
plt.show()
