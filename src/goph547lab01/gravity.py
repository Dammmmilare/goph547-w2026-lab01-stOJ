import numpy as np
import matplotlib.pyplot as plt

def gravity_potential_point(x, xm, m, G=6.674e-11) :

    """
    Compute the gravity potential due to a point mass.

    Parameters
    ---------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation.
        Default in SI units.
        Allows user to modify if using different unit.
    
    Returns
    ------
    float
            Gravity potential at x due to anomaly at xm.
    """

    """
    Function
    --------
    """
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)
    G = float(G)
    m = float(m)
    r = np.linalg.norm(x - xm)
    u = - G * m / r
    return u


def gravity_effect_point(x, xm, m, G=6.674e-11) :

    """
    Compute the vertical gravity effect due to a point mass (positive downward).

    Parameters
    ---------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation.
        Default in SI units.
        Allows user to modify if using different unit.

    Returns
    ------
    float
            Gravity effect at x due to anomaly at xm.
    """

    """
    Function
    --------
    """
    G = float(G)
    m = float(m)
    x = np.asarray(x, dtype=float)
    xm = np.asarray(xm, dtype=float)
    r = np.linalg.norm(x - xm)
    gz = G * m * (x[2] - xm[2]) / r**3
    return gz