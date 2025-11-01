import numpy as np

def genie_power(t_gyr):
    return 2.0 * np.log1p(t_gyr) + 1.5 * np.sin(2 * np.pi * t_gyr / 0.0966) + 3.0 * np.exp(-t_gyr / 0.18)

def aladin_vinfinity(r_pc, t_yr, M_sun=1e8):
    G = 6.6743e-11
    c = 3e8
    a0 = 1.2e-10
    alpha_A = 0.1
    tau_A = 80e6 * 365.25 * 86400
    rho = 1e-20
    JxB = 1e-12

    r = r_pc * 3.086e16
    t = t_yr * 365.25 * 86400
    M = M_sun * 1.989e30
    g_N = G * M / r**2
    plasma = 1 + alpha_A * abs(JxB) / (c * rho * r)
    t_gyr = t_yr / 1e9
    genie = genie_power(t_gyr)
    decay = np.exp(-t / tau_A)
    return np.sqrt(G * M / r) * np.sqrt(1 + a0 / g_N) * plasma * genie * decay

# Test
print(f"z=14: {aladin_vinfinity(100, 3e8, 1e8):.2e}")
print(f"z=20: {aladin_vinfinity(200, 1.5e8, 1e9):.2e}")
