import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point 

def main():
    
    # Parameters:
    m = 1.0e7  # mass of point anomaly in kg-m
    xm = np.array([0.0, 0.0, -10.0])  # location of point mass anomaly in m
    zp = [0.0, 10.0, 100.0]  # survey plane at z=0

    # Grid survey points at 25 and 5m spacing:
    x_25, y_25 = np.meshgrid(np.arange(-100, 125, 25), np.arange(-100, 125, 25))
    x_5, y_5 = np.meshgrid(np.arange(-100, 105, 5), np.arange(-100, 105, 5))
   

    # Allocate arrays for potential and gravity effect
    # survey grid at 25 m grid spacing
    u_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    #gz_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    gz_25 = np.zeros_like(u_25)
    
    xs_25 = x_25 #[0, :]
    ys_25 = y_25 #[:, 0]

    # survey grid at 5 m grid spacing
    u_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    # gz_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    gz_5 = np.zeros_like(u_5)
    xs_5 = x_5 # [0, :]
    ys_5 = y_5 # [:, 0]

    # computing potential and gravity effect at each survey point
    for k in range(len(zp)):
        for i in range(x_25.shape[0]):
            for j in range(x_25.shape[1]):
                x_survey = np.array([x_25[i, j], y_25[i, j], zp[k]])
                u_25[i, j, k] = gravity_potential_point(x_survey, xm, m)
                gz_25[i, j, k] = gravity_effect_point(x_survey, xm, m)

        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x_survey = np.array([x_5[i, j], y_5[i, j], zp[k]])
                u_5[i, j, k] = gravity_potential_point(x_survey, xm, m)
                gz_5[i, j, k] = gravity_effect_point(x_survey, xm, m)
    
# generating contour plots

    # generating contour plots for 25 m grid spacing        
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f'Single Mass Anomaly @ 25m grid spacing', fontsize=12)
        
    for k in range(len(zp)):

        c1 = axs[k, 0].contourf(xs_25, ys_25, u_25[:, :, k], levels=20, cmap='viridis', vmin=u_25.min(), vmax=u_25.max())
            
        axs[k, 0].plot(x_25, y_25, 'xk', markersize=2)
        axs[k, 0].set_title(f'U at z = {zp[k]} m')
        fig.colorbar(c1, ax=axs[k, 0])

        c2 = axs[k, 1].contourf(xs_25, ys_25, gz_25[:, :, k], levels=20, cmap='viridis', vmin=gz_25.min(), vmax=gz_25.max())
            
        axs[k, 1].plot(x_25, y_25, 'xk', markersize=2)
        axs[k, 1].set_title(f'gz at z = {zp[k]} m')
        fig.colorbar(c2, ax=axs[k, 1])

    plt.tight_layout()
    plt.savefig(f'examples/single_mass_anomaly_25m_grid_z_{zp[k]}m.png')
    plt.show()

    # generating contour plots for 5 m grid spacing
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f'Single Mass Anomaly @ 5m grid spacing', fontsize=12)
        
    for k in range(len(zp)):

        c1 = axs[k, 0].contourf(xs_5, ys_5, u_5[:, :, k], levels=20, cmap='viridis', vmin=u_5.min(), vmax=u_5.max())
            
        axs[k, 0].plot(x_5, y_5, 'xk', markersize=2)
        axs[k, 0].set_title(f'U at z = {zp[k]} m')
        fig.colorbar(c1, ax=axs[k, 0])

        c2 = axs[k, 1].contourf(xs_5, ys_5, gz_5[:, :, k], levels=20, cmap='viridis', vmin=gz_5.min(), vmax=gz_5.max())
            
        axs[k, 1].plot(x_5, y_5, 'xk', markersize=2)
        axs[k, 1].set_title(f'gz at z = {zp[k]} m')
        fig.colorbar(c2, ax=axs[k, 1])

    plt.tight_layout()
    plt.savefig(f'examples/single_mass_anomaly_5m_grid_z_{zp[k]}m.png')
    plt.show()

if __name__ == "__main__":
    main()