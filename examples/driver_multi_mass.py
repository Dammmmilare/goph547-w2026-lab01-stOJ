import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (loadmat, savemat)
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def generate_mass_anomaly_sets():
    
    # Generate and save mass anomaly data sets
    os.makedirs("examples", exist_ok=True)
   
    m = 1.0e7 
    cen = np.array([0.0, 0.0, -10.0])  # cen is the centroid.

    for i in range(3):

        masses =  np.zeros(5)
        locations = np.zeros((5, 3))

        for j in range(4):
            masses[j] = np.random.normal(m/5, m/100)

            locations[j, 0] = np.random.normal(0.0, 20.0)
            locations[j, 1] = np.random.normal(0.0, 20.0)
            locations[j, 2] = np.random.normal(-10.0, 2.0)

        masses[4] = m - np.sum(masses[:4])

        weighted = np.sum(masses[:4, None]* locations[:4], axis=0)
        locations[4] = (m * cen - weighted) / masses[4]

        savemat(f"examples/mass_set_{i:02d}.mat", {"masses": masses, "locations": locations})

    print ("Mass anomaly data sets generated and saved in 'example' directory.")
    
    

def load_mass_anomaly_set(set_id):

    data = loadmat(f"examples/mass_set_{set_id:02d}.mat")

    masses = data["masses"].flatten()
    locations = data["locations"]

    return  masses, locations

# Parameters:
    
m = 1.0e7  # mass of point anomaly in kg-m
zp = [0.0, 10.0, 100.0]  # survey plane at z=0.0

x25, y25 = np.meshgrid(np.arange(-100, 125, 25), np.arange(-100, 125, 25))
x5, y5 = np.meshgrid(np.arange(-100, 105, 5), np.arange(-100, 105, 5))


def compute_fields(xg, yg, masses, locations):

    u = np.zeros((xg.shape[0], xg.shape[1], len(zp)))
    gz = np.zeros_like(u)

    for k in range(len(zp)):
        for i in range(xg.shape[0]):
            for j in range(xg.shape[1]):

                x_survey = np.array([xg[i, j], yg[i, j], zp[k]])

                for m, loc in zip(masses, locations):
                    u[i, j, k] += gravity_potential_point(x_survey, loc, m)
                    gz[i, j, k] += gravity_effect_point(x_survey, loc, m)

    return u, gz

def plot_fields(xg, yg, u, gz, zp, grid_spacing):

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f'Multiple Mass Anomalies @ {grid_spacing}m grid spacing', fontsize=12)
        
    for k in range(len(zp)):

        c1 = axs[k, 0].contourf(xg, yg, u[:, :, k], levels=20, cmap='viridis', vmin=u.min(), vmax=u.max())
            
        axs[k, 0].plot(xg, yg, 'xk', markersize=2)
        axs[k, 0].set_title(f'U at z = {zp[k]} m')

        c2 = axs[k, 1].contourf(xg, yg, gz[:, :, k], levels=20, cmap='viridis', vmin=gz.min(), vmax=gz.max())
            
        axs[k, 1].plot(xg, yg, 'xk', markersize=2)
        axs[k, 1].set_title(f'gz at z = {zp[k]} m')

    fig.colorbar(c1, ax=axs[:, 0], orientation='vertical', label='Gravity Potential (J/kg)')
    fig.colorbar(c2, ax=axs[:, 1], orientation='vertical', label='Gravity Effect (m/s^2)')

    plt.tight_layout()
    plt.savefig(f"examples/mass_anomalies_{grid_spacing}m.png", dpi=300)
    plt.show()
    plt.close(fig)
    
def main():

    if not os.path.exists("examples/mass_set_00.mat"):
        print("Generating mass anomaly data sets...")
        generate_mass_anomaly_sets()

    for i in range(3):

        print(f"Processing mass anomaly set {i}...")
        
        masses, locations = load_mass_anomaly_set(i)

        u25, gz25 = compute_fields(x25, y25, masses, locations)
        plot_fields(x25, y25, u25, gz25, zp, 25)

        u5, gz5 = compute_fields(x5, y5, masses, locations)
        plot_fields(x5, y5, u5, gz5, zp, 5)

if __name__ == "__main__":
    main()