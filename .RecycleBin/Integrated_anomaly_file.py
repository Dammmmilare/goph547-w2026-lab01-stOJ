import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

from src.goph547lab01.gravity import (
    gravity_potential_point,
    gravity_effect_point,
)

VOXEL_SIZE = 2.0
dV = VOXEL_SIZE**3


# ===============================
# 1. Load anomaly
# ===============================
"""def load_anomaly():
    if not os.path.exists("anomaly_data.mat"):
        raise FileNotFoundError("Missing file anomaly_data.mat")

    data = loadmat("anomaly_data.mat")
    return data["x"], data["y"], data["z"], data["rho"]"""

def load_anomaly():

    script_dir = pathlib.Path(__file__).resolve().parent
    data_path = script_dir / "anomaly_data.mat"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing file {data_path}")

    data = loadmat(data_path)

    return data["x"], data["y"], data["z"], data["rho"]


# ===============================
# 2. Mass properties
# ===============================
def compute_mass_properties(x, y, z, rho):

    mm = rho * dV
    total_mass = np.sum(mm)

    x_bar = np.sum(x * mm) / total_mass
    y_bar = np.sum(y * mm) / total_mass
    z_bar = np.sum(z * mm) / total_mass

    rho_max = np.max(rho)
    rho_mean = np.mean(rho)

    print("\n===== MASS PROPERTIES =====")
    print(f"Total mass: {total_mass:.3e} kg")
    print(f"Barycentre: [{x_bar:.3f}, {y_bar:.3f}, {z_bar:.3f}] m")
    print(f"Max density: {rho_max:.3e} kg/m^3")
    print(f"Mean density: {rho_mean:.3e} kg/m^3")

    return mm, np.array([x_bar, y_bar, z_bar])


# ===============================
# 3. Extract official subregion
# ===============================
def extract_subregion(x, y, z, rho, mm):

    kx_min, kx_max = 40, 60
    ky_min, ky_max = 44, 56
    kz_min, kz_max = 7, 13

    xmin = x[0, kx_min, 0]
    xmax = x[0, kx_max, 0]
    ymin = y[ky_min, 0, 0]
    ymax = y[ky_max, 0, 0]
    zmin = z[0, 0, kz_min]
    zmax = z[0, 0, kz_max]

    print("\n===== SUBREGION =====")
    print("x range:", xmin, xmax)
    print("y range:", ymin, ymax)
    print("z range:", zmin, zmax)

    mm_sub = mm[ky_min:ky_max+1,
                kx_min:kx_max+1,
                kz_min:kz_max+1].flatten()

    xm_sub = x[ky_min:ky_max+1,
               kx_min:kx_max+1,
               kz_min:kz_max+1].flatten()

    ym_sub = y[ky_min:ky_max+1,
               kx_min:kx_max+1,
               kz_min:kz_max+1].flatten()

    zm_sub = z[ky_min:ky_max+1,
               kx_min:kx_max+1,
               kz_min:kz_max+1].flatten()

    return mm_sub, xm_sub, ym_sub, zm_sub


# ===============================
# 4. Density cross-sections
# ===============================
def plot_density(x, y, z, rho, bary):

    x_bar, y_bar, z_bar = bary

    rho_bar_min = 0.0
    rho_bar_max = 0.6

    plt.figure(figsize=(8, 9))

    # Mean along y
    plt.subplot(3, 1, 1)
    plt.contourf(
        x[0, :, :],
        z[0, :, :],
        np.mean(rho, axis=0),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(x_bar, z_bar, "xk")
    plt.colorbar(label=r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title("Mean density along y-axis")

    # Mean along x
    plt.subplot(3, 1, 2)
    plt.contourf(
        y[:, 0, :],
        z[:, 0, :],
        np.mean(rho, axis=1),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(y_bar, z_bar, "xk")
    plt.colorbar(label=r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.title("Mean density along x-axis")

    # Mean along z
    plt.subplot(3, 1, 3)
    plt.contourf(
        x[:, :, 0],
        y[:, :, 0],
        np.mean(rho, axis=2),
        cmap="viridis_r",
        levels=np.linspace(rho_bar_min, rho_bar_max, 200),
    )
    plt.plot(x_bar, y_bar, "xk")
    plt.colorbar(label=r"$\bar{\rho}$ [$kg/m^3$]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Mean density along z-axis")

    plt.tight_layout()
    plt.savefig("anomaly_mean_density.png", dpi=300)
    plt.close()


# ===============================
# 5. Generate survey data
# ===============================
def generate_survey(mm_sub, xm_sub, ym_sub, zm_sub):

    x_5, y_5 = np.meshgrid(
        np.linspace(-100, 100, 41),
        np.linspace(-100, 100, 41)
    )
    zp = [0.0, 1.0, 100.0, 110.0]

    U_5 = np.zeros((41, 41, 4))
    g_5 = np.zeros((41, 41, 4))

    for mm_k, xx_k, yy_k, zz_k in zip(mm_sub, xm_sub, ym_sub, zm_sub):

        xm_k = [xx_k, yy_k, zz_k]

        for k, zz in enumerate(zp):
            for i in range(41):
                for j in range(41):
                    x_obs = [x_5[i, j], y_5[i, j], zz]

                    U_5[i, j, k] += gravity_potential_point(
                        x_obs, xm_k, mm_k
                    )

                    g_5[i, j, k] += gravity_effect_point(
                        x_obs, xm_k, mm_k
                    )

    savemat(
        "anomaly_survey_data.mat",
        {"x_5": x_5,
         "y_5": y_5,
         "zp": zp,
         "g_5": g_5,
         "U_5": U_5}
    )

    print("Survey data generated.")


# ===============================
# 6. Compute derivatives
# ===============================
def compute_derivatives(x_5, y_5, zp, g_5):

    dx = x_5[0, 1] - x_5[0, 0]
    dy = y_5[1, 0] - y_5[0, 0]

    dgdz = np.stack((
        (g_5[:, :, 1] - g_5[:, :, 0]) / (zp[1] - zp[0]),
        (g_5[:, :, 3] - g_5[:, :, 2]) / (zp[3] - zp[2]),
    ), axis=-1)

    d2gdz2 = np.stack((
        -(g_5[2:,1:-1,0] - 2*g_5[1:-1,1:-1,0] + g_5[:-2,1:-1,0]) / dy**2
        -(g_5[1:-1,2:,0] - 2*g_5[1:-1,1:-1,0] + g_5[1:-1,:-2,0]) / dx**2,

        -(g_5[2:,1:-1,2] - 2*g_5[1:-1,1:-1,2] + g_5[:-2,1:-1,2]) / dy**2
        -(g_5[1:-1,2:,2] - 2*g_5[1:-1,1:-1,2] + g_5[1:-1,:-2,2]) / dx**2,
    ), axis=-1)

    return dgdz, d2gdz2


# ===============================
# 7. Main
# ===============================
def main():

    x, y, z, rho = load_anomaly()

    mm, bary = compute_mass_properties(x, y, z, rho)

    mm_sub, xm_sub, ym_sub, zm_sub = extract_subregion(
        x, y, z, rho, mm
    )

    plot_density(x, y, z, rho, bary)

    if not os.path.exists("anomaly_survey_data.mat"):
        generate_survey(mm_sub, xm_sub, ym_sub, zm_sub)

    survey = loadmat("anomaly_survey_data.mat")

    x_5 = survey["x_5"]
    y_5 = survey["y_5"]
    zp = survey["zp"][0]
    g_5 = survey["g_5"]

    dgdz, d2gdz2 = compute_derivatives(x_5, y_5, zp, g_5)

    print("\nAll results successfully generated.")


if __name__ == "__main__":
    main()