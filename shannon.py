"""
Shannon information
===================
"""

import numpy as np


def information(file_name, ln_z):
    """
    @returns Shannon information in chain
    """
    weight, chi_sq = np.loadtxt(file_name, unpack=True)[:2]
    mean_chi_sq = np.dot(weight, chi_sq)
    mean_ln_like = -0.5 * mean_chi_sq
    return mean_ln_like - ln_z


if __name__ == "__main__":

    import sys

    FILE_NAME = sys.argv[1]
    LN_Z = sys.argv[2]
    INFO = information(FILE_NAME, LN_Z)

    print "information  = {}".format(INFO)
