import numpy as np
import matplotlib.pyplot as plt
from src.gravity import gravity_potential_point, gravity_effect_point 

# Single Mass Anomaly & Contour Plot

def main():
    # Define parameters for single mass anomaly
    m = 1e7 # mass in kg
    xm = np.array([0.0, 0.0, -100.0]) (m) # location of mass anomaly (x,y,z) in meters

# type: ignore