import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from src.goph547lab01.gravity import gravity_effect_point

VOXEL_SIZE = 2.0
dV = VOXEL_SIZE**3 
SURVEY_SPACING = 5.0

def load_anomaly():

    data = loadmat("examples/anomaly_data.mat")

    x = data["x"]
    y = data["y"]
    z = data["z"]
    rho = data["rho"]

    return x, y, z, rho

def compute_mass_properties(x, y, z, rho):

    total_mass = np.sum(rho) * dV

    x_bar = np.sum(rho * x) * dV / total_mass
    y_bar = np.sum(rho * y) * dV / total_mass
    z_bar = np.sum(rho * z) * dV / total_mass

    rho_max = np.max(rho)
    rho_mean = np.mean(rho)

    print("\n===== MASS PROPERTIES =====")
    print("Total mass:", total_mass)
    print("Barycentre (x,y,z):", x_bar, y_bar, z_bar)
    print("Maximum density:", rho_max)
    print("Mean density:", rho_mean)

    return total_mass, np.array([x_bar, y_bar, z_bar]), rho_mean

def plot_density_cross_sections(x, y, z, rho, bary):

    rho_xz = np.mean(rho, axis=1)
    rho_yz = np.mean(rho, axis=0)
    rho_xy = np.mean(rho, axis=2)

    vmin = rho.min()
    vmax = rho.max()

    xb, yb, zb = bary

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    # xz
    c1 = axs[0].contourf(x[:,0,:], z[:,0,:], rho_xz,
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0].plot(xb, zb, "xk", markersize=3)
    axs[0].set_title("Mean Density - XZ Plane")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("z (m)")
    plt.colorbar(c1, ax=axs[0])

    # yz
    c2 = axs[1].contourf(y[0,:,:], z[0,:,:], rho_yz,
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axs[1].plot(yb, zb, "xk", markersize=3)
    axs[1].set_title("Mean Density - YZ Plane")
    axs[1].set_xlabel("y (m)")
    axs[1].set_ylabel("z (m)")
    plt.colorbar(c2, ax=axs[1])

    # xy
    c3 = axs[2].contourf(x[:,:,0], y[:,:,0], rho_xy,
                         cmap="viridis", vmin=vmin, vmax=vmax)
    axs[2].plot(xb, yb, "xk", markersize=3)
    axs[2].set_title("Mean Density - XY Plane")
    axs[2].set_xlabel("x (m)")
    axs[2].set_ylabel("y (m)")
    plt.colorbar(c3, ax=axs[2])

    plt.tight_layout()
    plt.show()

def analyze_dense_region(x, y, z, rho, overall_mean):

    threshold = 0.1 * rho.max()
    mask = rho > threshold

    mean_region = np.mean(rho[mask])

    print("\n===== NON-NEGLIGIBLE REGION =====")
    print("Threshold used:", threshold)
    print("x range:", x[mask].min(), x[mask].max())
    print("y range:", y[mask].min(), y[mask].max())
    print("z range:", z[mask].min(), z[mask].max())
    print("Mean density in region:", mean_region)
    print("Increase relative to overall mean:",
          mean_region - overall_mean)

def forward_model_gz(x, y, z, rho, survey_z):

    xg, yg = np.meshgrid(
        np.arange(x.min(), x.max(), SURVEY_SPACING),
        np.arange(y.min(), y.max(), SURVEY_SPACING)
    )

    gz = np.zeros_like(xg)

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    rho_flat = rho.flatten()

    for i in range(xg.shape[0]):
        for j in range(xg.shape[1]):

            survey_point = np.array([xg[i,j], yg[i,j], survey_z])

            for xv, yv, zv, rhov in zip(x_flat, y_flat, z_flat, rho_flat):

                mass = rhov * dV
                source = np.array([xv, yv, zv])

                gz[i,j] += gravity_effect_point(
                    survey_point, source, mass
                )

    return xg, yg, gz

def first_vertical_derivative(gz_lower, gz_upper, dz):
    return (gz_upper - gz_lower) / dz

def second_vertical_derivative(gz, dx):

    d2x = (np.roll(gz, -1, axis=1) - 2*gz + np.roll(gz, 1, axis=1)) / dx**2
    d2y = (np.roll(gz, -1, axis=0) - 2*gz + np.roll(gz, 1, axis=0)) / dx**2

    return -(d2x + d2y)

def main():

    x, y, z, rho = load_anomaly()

    total_mass, bary, overall_mean = compute_mass_properties(x, y, z, rho)

    plot_density_cross_sections(x, y, z, rho, bary)

    analyze_dense_region(x, y, z, rho, overall_mean)

    print("\nRunning forward modelling... (may take time)")

    xg, yg, gz_0 = forward_model_gz(x, y, z, rho, 0.0)
    _, _, gz_1 = forward_model_gz(x, y, z, rho, 1.0)
    _, _, gz_100 = forward_model_gz(x, y, z, rho, 100.0)
    _, _, gz_110 = forward_model_gz(x, y, z, rho, 110.0)

    dgz_dz_0 = first_vertical_derivative(gz_0, gz_1, 1.0)
    dgz_dz_100 = first_vertical_derivative(gz_100, gz_110, 10.0)

    d2gz_dz2_0 = second_vertical_derivative(gz_0, SURVEY_SPACING)
    d2gz_dz2_100 = second_vertical_derivative(gz_100, SURVEY_SPACING)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    vmin = min(gz_0.min(), gz_1.min(), gz_100.min(), gz_110.min())
    vmax = max(gz_0.max(), gz_1.max(), gz_100.max(), gz_110.max())

    axs[0,0].contourf(xg, yg, gz_0, vmin=vmin, vmax=vmax)
    axs[0,0].set_title("gz @ 0 m")

    axs[0,1].contourf(xg, yg, gz_100, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("gz @ 100 m")

    axs[1,0].contourf(xg, yg, gz_1, vmin=vmin, vmax=vmax)
    axs[1,0].set_title("gz @ 1 m")

    axs[1,1].contourf(xg, yg, gz_110, vmin=vmin, vmax=vmax)
    axs[1,1].set_title("gz @ 110 m")

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].contourf(xg, yg, d2gz_dz2_0)
    axs[0].set_title("∂²gz/∂z² @ 0 m")

    axs[1].contourf(xg, yg, d2gz_dz2_100)
    axs[1].set_title("∂²gz/∂z² @ 100 m")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()