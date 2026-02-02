import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point 

# Set up for the Single Mass Anomaly & Contour Plot

def main():
    
    m = 1.0e7  # mass of point anomaly in kg-m
    xm = np.array([0.0, 0.0, -10.0])  # location of point mass anomaly in m

    # Create a grid of survey points at z=0
    x_25, y_25 = np.meshgrid(np.linspace(-100, 100, 9), np.linspace(-100, 100, 9))
    x_5, y_5 = np.meshgrid(np.linspace(-100, 100, 41), np.linspace(-100, 100, 41))
    zp = [0.0, 10.0, 100.0]  # survey plane at z=0

    # Stating survey grids  @ 25 m and 5 m spacing.
    # survey grid at 25 m grid spacing
    u_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    gz_25 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))
    xs_25 = x_25[0, :]
    ys_25 = y_25[:, 0]

    # survey grid at 5 m grid spacing
    u_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    gz_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))
    xs_5 = x_5[0, :]
    ys_5 = y_5[:, 0]

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
    
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(xs_25, ys_25, gz_25[:, :, k], levels=20)
        plt.title(f'Gravity anomaly at z={zp[k]}m (25m grid)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar()
        plt.savefig(f'gravity_anomaly_z_{zp[k]}m (25m grid).png')

        plt.subplot(1, 2, 2)
        plt.contourf(xs_5, ys_5, gz_5[:, :, k], levels=20)
        plt.title(f'Gravity anomaly at z={zp[k]}m (5m grid)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'gravity_anomaly_z_{zp[k]}m (5m grid).png')
        

if __name__ == "__main__":
    main()