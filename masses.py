"""
Neutralino and chargino masses
==============================
"""

import numpy as np


MZ = 91.1876
MW = 80.385
CW = MW / MZ
SW = (1. - CW**2)**0.5


def neutralino(M1, M2, mu, tan_beta):
    """
    @returns Absolute values of neturalino masses in ascending order
    """
    beta = np.arctan(tan_beta)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)

    M = np.array([[M1, 0., - cos_beta * SW * MZ, sin_beta * SW * MZ],
                  [0., M2, cos_beta * CW * MZ, - sin_beta * CW * MZ],
                  [- cos_beta * SW * MZ, cos_beta * CW * MZ, 0., -mu],
                  [sin_beta * SW * MZ, - sin_beta * CW * MZ, -mu, 0.]])
    assert np.all(M == M.T)
    mass = abs(np.linalg.eigvalsh(M))
    return np.sort(mass)

def chargino(M2, mu, tan_beta):
    """
    @returns Absolute values of chargino masses in ascending order
    """
    beta = np.arctan(tan_beta)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)

    X = np.array([[M2, 2**0.5 * MW * sin_beta],
                  [2**0.5 * MW *cos_beta, mu]])

    mass_squared = np.linalg.eigvalsh(np.dot(X, X.T))
    mass = abs(mass_squared)**0.5
    return np.sort(mass)

def masses(M1, M2, mu, tan_beta):
    """
    @returns Absolute values of neturalino masses in ascending order, followed
    by absolute values of chargino masses in ascending order
    """
    return np.append(neutralino(M1, M2, mu, tan_beta), chargino(M2, mu, tan_beta))


if __name__ == "__main__":

    print masses(10., 100., 50., 4.)
