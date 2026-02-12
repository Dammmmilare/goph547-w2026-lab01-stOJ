import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import (loadmat, savemat)
from src.goph547lab01.gravity import gravity_potential_point, gravity_effect_point

