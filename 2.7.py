import numpy as np
import matplotlib.pyplot as plt
import time

print("🌌 Plasma Cosmology v2.7 — Fixed EMF Averaging + Stable CT")

# ====================== GRID ======================
N = 72
L = 48.0
dx = L / N
x = y = z = np.linspace(-L/2, L/2, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

r_cyl = np.sqrt(X**2 + Y**2)
r_sph = np.sqrt(X**2 + Y**2 + Z**2 + 1e-8)

mu0 = 4 * np.pi * 1e-7
eta = 0.0028
nu = 0.016
dt = 0.00012
steps = 500
max_v = 300 * 1000.0

# ====================== STAGGERED FIELDS ======================
Bx = np.zeros((N+1, N, N), dtype=np.float32)
By = np.zeros((N, N+1, N), dtype=np.float32)
Bz = np.zeros((N, N, N+1), dtype=np.float32)

rho = np.zeros((N, N, N), dtype=np.float32)
vx = np.zeros((N, N, N), dtype=np.float32)
vy = np.zeros((N, N, N), dtype=np.float32)
vz = np.zeros((N, N, N), dtype=np.float32)
p = np.zeros((N, N, N), dtype=np.float32)

# ====================== INITIAL CONDITIONS ======================
rho[...] = 3.2e-24 * np.exp(-r_cyl / 13.0) * np.exp(-np.abs(Z)/5.0)
J_z_cell = 8.0e17 * np.exp(-r_cyl**2 / (2*7.8**2))

vx[...] = 0.0
vy[...] = 0.0
vz[...] = 0.0

p[...] = 9e-13 * rho

# Seed B on faces
for k in range(N+1):
    z_face = -L/2 + k * dx
    r_slice = np.sqrt(X[:,:,0]**2 + Y[:,:,0]**2)
    Bz[:,:,k] = 7.5e-10 * np.exp(-(r_slice**2 + z_face**2) / (14.0**2))

# NFW
M_vir = 1.2e12 * 1.989e30
rs = 20.0 * 3.086e19
def nfw_mass(r):
    x = r / rs
    m = np.log(1 + x) - x / (1 + x)
    return M_vir * (m / (np.log(2) - 0.5))
M_enc_dm = nfw_mass(r_sph)

print("Starting v2.7 with fixed EMF averaging...\n")

start = time.time()

for step in range(steps):
    # Edge-centered EMFs with proper averaging
    Ex = np.zeros((N, N+1, N+1), dtype=np.float32)
    Ey = np.zeros((N+1, N, N+1), dtype=np.float32)
    Ez = np.zeros((N+1, N+1, N), dtype=np.float32)

    # Ex (y-z edges)
    vy_avg = 0.25 * (vy[:, :-1, :-1] + vy[:, 1:, :-1] + vy[:, :-1, 1:] + vy[:, 1:, 1:])
    vz_avg = 0.25 * (vz[:, :-1, :-1] + vz[:, 1:, :-1] + vz[:, :-1, 1:] + vz[:, 1:, 1:])
    By_avg = 0.5 * (By[:, :-1, :] + By[:, 1:, :])
    Bz_avg = 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
    Ex[:, 1:, 1:] = -(vy_avg * Bz_avg - vz_avg * By_avg)

    # Ey (x-z edges)
    vz_avg = 0.25 * (vz[:-1, :, :-1] + vz[1:, :, :-1] + vz[:-1, :, 1:] + vz[1:, :, 1:])
    vx_avg = 0.25 * (vx[:-1, :, :-1] + vx[1:, :, :-1] + vx[:-1, :, 1:] + vx[1:, :, 1:])
    Bx_avg = 0.5 * (Bx[:-1, :, :] + Bx[1:, :, :])
    Bz_avg = 0.5 * (Bz[:, :, :-1] + Bz[:, :, 1:])
    Ey[1:, :, 1:] = -(vz_avg * Bx_avg - vx_avg * Bz_avg)

    # Ez (x-y edges)
    vx_avg = 0.25 * (vx[:-1, :-1, :] + vx[1:, :-1, :] + vx[:-1, 1:, :] + vx[1:, 1:, :])
    vy_avg = 0.25 * (vy[:-1, :-1, :] + vy[1:, :-1, :] + vy[:-1, 1:, :] + vy[1:, 1:, :])
    Bx_avg = 0.5 * (Bx[:-1, :, :] + Bx[1:, :, :])
    By_avg = 0.5 * (By[:, :-1, :] + By[:, 1:, :])
    Ez[1:, 1:, :] = -(vx_avg * By_avg - vy_avg * Bx_avg)

    # Update B using circulation (true CT)
    Bx[1:-1] += dt * ((Ez[:,1:,1:] - Ez[:,:-1,1:]) - (Ey[:,1:,1:] - Ey[:,1:,:-1])) / dx
    By[:,1:-1] += dt * ((Ex[:,1:,1:] - Ex[:,1:,:-1]) - (Ez[1:,1:,1:] - Ez[:-1,1:,1:])) / dx
    Bz[:,:,1:-1] += dt * ((Ey[1:,1:,1:] - Ey[:-1,1:,1:]) - (Ex[:,1:,1:] - Ex[:,:-1,1:])) / dx

    # Cell-centered B
    Bx_c = (Bx[:-1] + Bx[1:]) / 2
    By_c = (By[:,:-1] + By[:,1:]) / 2
    Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2

    # Lorentz force
    Jx = (np.gradient(Bz_c, dx, axis=1) - np.gradient(By_c, dx, axis=2)) / mu0
    Jy = (np.gradient(Bx_c, dx, axis=2) - np.gradient(Bz_c, dx, axis=0)) / mu0
    Jz_total = J_z_cell + (np.gradient(By_c, dx, axis=0) - np.gradient(Bx_c, dx, axis=1)) / mu0

    Fx = Jy * Bz_c - Jz_total * By_c
    Fy = Jz_total * Bx_c - Jx * Bz_c
    Fz = Jx * By_c - Jy * Bx_c

    # Gravity + Pressure + Viscosity
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

    divv = (np.gradient(vx, dx, axis=0) + np.gradient(vy, dx, axis=1) + np.gradient(vz, dx, axis=2))
    Fx -= nu * divv * vx
    Fy -= nu * divv * vy
    Fz -= nu * divv * vz

    # Update velocities
    vx += dt * Fx / (rho + 1e-30)
    vy += dt * Fy / (rho + 1e-30)
    vz += dt * Fz / (rho + 1e-30)

    v_tot = np.sqrt(vx**2 + vy**2 + vz**2)
    mask = v_tot > max_v
    if np.any(mask):
        scale = max_v / (v_tot[mask] + 1e-30)
        vx[mask] *= scale
        vy[mask] *= scale
        vz[mask] *= scale

    # Continuity
    div_v = (np.gradient(rho*vx, dx, axis=0) + np.gradient(rho*vy, dx, axis=1) + np.gradient(rho*vz, dx, axis=2))
    rho += dt * (-div_v)
    rho = np.maximum(rho, 5e-28)

    if step % 80 == 0 or step == steps-1:
        Bmag = np.sqrt(Bx_c**2 + By_c**2 + Bz_c**2).max() * 1e6
        vmax = v_tot.max() / 1000
        print(f"Step {step:4d} | Bmax = {Bmag:.2f} μG | vmax = {vmax:.1f} km/s")

print(f"Finished in {time.time() - start:.1f} s")

# ====================== FINAL PLOTS ======================
mid = N // 2
Bx_c = (Bx[:-1] + Bx[1:]) / 2
By_c = (By[:,:-1] + By[:,1:]) / 2
Bz_c = (Bz[:,:,:-1] + Bz[:,:,1:]) / 2

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.sqrt(Bx_c[:,:,mid]**2 + By_c[:,:,mid]**2), cmap='magma')
plt.title('|B| mid-plane')
plt.colorbar()

plt.subplot(1, 2, 2)
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

plt.suptitle('Plasma Cosmology v2.6 — Fixed CT with Proper Indexing')
plt.tight_layout()
plt.show()
