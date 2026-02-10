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
    m = 1.0e7  # mass of point anomaly in kg-m.
    xm = np.array([0.0, 0.0, -10.0])  # location of point mass anomaly in m.
    zp = [0.0, 10.0, 100.0]  # survey plane at z=0.0.

    for i in range(x_25.shape[0]):
        m[i] = np.random.normal(m / xm, 0.1 * m / xm)

        xm[i, 0] = np.random.normal(0.0, 20.0)  # x-coordinate of mass anomaly location in m.
        xm[i, 1] = np.random.normal(0.0, 20.0)  # y-coordinate of mass anomaly location in m.
        xm[i, 2] = np.random.normal(-10.0, 2.0)  # z-coordinate of mass anomaly location in m.

    m [4] = m - np.sum(m[:4])  # Adjust the last mass to ensure total mass is correct  

    weighted_xm = np.sum(m[:, np.newaxis] * xm, axis=0) / np.sum(m)
    return m, xm, zp

if __name__ == "__main__":
    main()