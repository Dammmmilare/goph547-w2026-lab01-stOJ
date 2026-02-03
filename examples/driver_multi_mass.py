import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (loadmat, savemat)
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point

def generate_mass_anomaly_sets():
        # Run check to see if the mass anomalies have been generated, if not the script will generate them.
    
    if (not os.path.exists("example/mass_set_0.mat") or
        not os.path.exists("example/mass_set_1.mat") or
        not os.path.exists("example/mass_set_2.mat")): 
        generate_mass_anomaly_sets() 

def main():

    # Setting survey grid parameters.
    x_25, y_25 = np.meshgrid(np.linspace(-100, 100, 9), np.linspace(-100, 100, 9))
    x_5, y_5 = np.meshgrid(np.linspace(-100, 100, 41), np.linspace(-100, 100, 41))
    zp = [0.0, 10.0, 100.0]

    # loading data sets.

    # stating multiple survey grids @ 25 m and 5 m spacing.
    
if __name__ == "__main__":
    main()