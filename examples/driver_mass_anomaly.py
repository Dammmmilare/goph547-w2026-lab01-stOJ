import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (loadmat, savemat)
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def load_anomaly_data():

    data = loadmat("examples/anomaly_data.mat")
   
    x = data["x"]
    y = data["y"]
    z = data["z"]
    rho = data["rho"]

    return x, y, z, rho


def compute_mass_properties(x, y, z, rho):
    
    dv = 8
    total_mass = np.sum(rho) * dv

    x_bary = np.sum(x * rho) * dv / total_mass
    y_bary = np.sum(y * rho) * dv / total_mass 
    z_bary = np.sum(z * rho) * dv / total_mass

    rho_max = np.max(rho)
    rho_mean = np.mean(rho)

    print("Total Mass: ", total_mass)
    print("Barycenter: ", x_bary, y_bary, z_bary)
    print("Max Density: ", rho_max)
    print("Mean Density: ", rho_mean)

    return total_mass, np.array([x_bary, y_bary, z_bary])

def plot_density_cross_sections(x, y, z, rho, barycenter):
    
    rho_xz = np.mean(rho, axis=1)
    rho_yz = np.mean(rho, axis=0)  
    rho_xy = np.mean(rho, axis=2)

    vmin = rho.min()
    vmax = rho.max()

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))

    x_bary, y_bary, z_bary = barycenter

    c1 = axs[0].contourf(x[:,0,:], z[:,0,:], rho_xz, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].plot(x_bary, z_bary, 'ro', label='Barycenter', markersize=8)
    axs[0].set_title('Mean Density XZ Plane')
    plt.colorbar(c1, ax=axs[0])

    c2 = axs[1].contourf(y[0,:,:], z[0,:,:], rho_yz, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].plot(x_bary, z_bary, 'ro', label='Barycenter', markersize=8)
    axs[1].set_title('Mean Density YZ Plane')
    plt.colorbar(c2, ax=axs[1])

    c3 = axs[2].contourf(x[:,:,0], y[:,:,0], rho_xy, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[2].plot(x_bary, y_bary, 'ro', label='Barycenter', markersize=8)
    axs[2].set_title('Mean Density XY Plane')
    plt.colorbar(c3, ax=axs[2])

    plt.tight_layout()
    plt.show()

def forward_modelling_gz(x, y, z, rho, observation_points):
    
    xg, yg = np.meshgrid(np.arange(x.min(), x.max(), 50), np.arange(y.min(), y.max(), 50))
    
    gz = np.zeros_like(xg)

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    rho_flat = rho.flatten()
    
    for i in range(xg.shape[0]):
        for j in range(xg.shape[1]):
            obs_point = np.array([xg[i,j], yg[i,j], observation_points[2,0]])
            gz[i,j] = np.sum(gravity_effect_point(obs_point, x_flat, y_flat, z_flat, rho_flat)[:,2])

    return xg, yg, gz


