import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (loadmat, savemat)
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point


# Load anolamy data sets from .mat files

data = loadmat("examples/anomaly_data.mat")

def generate_mass_anomaly_sets():
    
   # Generate and save mass anomaly data sets

    os.makedirs("examples", exist_ok=True)
   
    savemat("examples/mass_set_0.mat", {"set_id": 0})
    savemat("examples/mass_set_1.mat", {"set_id": 1})
    savemat("examples/mass_set_2.mat", {"set_id": 2})  
    
    print ("Mass anomaly data sets generated and saved in 'example' directory.")

def main():

    if (not os.path.exists("examples/mass_set_0.mat") or
            not os.path.exists("examples/mass_set_1.mat") or
            not os.path.exists("examples/mass_set_2.mat")): 
            generate_mass_anomaly_sets() 

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

    # loading data sets.

    # stating multiple survey grids @ 25 m and 5 m spacing.
    
if __name__ == "__main__":
    main()