"""
SOFTSUSY
========
"""

import subprocess
from warnings import warn
import numpy as np
import pyslha
import tree_level


PROGRAM = "/home/andrew/codes/softsusy-4.1.6/softpoint.x"
KEYS = [1000022, 1000023, 1000025, 1000035, 1000024, 1000037]

SLHA_TEMPLATE = """
Block MODSEL		     # Select model
    1    0
Block SMINPUTS		     # Standard Model inputs
    1	1.279340000e+02	     # alpha^(-1) SM MSbar(MZ)
    2   1.166370000e-05	     # G_Fermi
    3   1.172000000e-01	     # alpha_s(MZ) SM MSbar
    4   9.118760000e+01	     # MZ(pole)
    5	4.250000000e+00	     # mb(mb) SM MSbar
    6   1.743000000e+02	     # mtop(pole)
    7	1.777000000e+00	     # mtau(pole)
BLOCK MINPAR
    3  {3}  # tanbeta(mZ)^DRbar
BLOCK EXTPAR
    0  3.e3    # Q
    1  {0}     # bino mass parameter M1
    2  {1}     # wino mass parameter M2
    3  5.e3    # gluino mass parameter M3
    23  {2}    # mu parameter
    26  5.e3   # mA pole mass
    31  3.e3   # sqrt(ml2(1,1))
    32  3.e3   # sqrt(ml2(2,2))
    33  3.e3   # sqrt(ml2(3,3))
    34  3.e3   # sqrt(me2(1,1))
    35  3.e3   # sqrt(me2(2,2))
    36  3.e3   # sqrt(me2(3,3))
    41  3.e3   # sqrt(mq2(1,1))
    42  3.e3   # sqrt(mq2(2,2))
    43  3.e3   # sqrt(mq2(3,3))
    44  3.e3   # sqrt(mu2(1,1))
    45  3.e3   # sqrt(mu2(2,2))
    46  3.e3   # sqrt(mu2(3,3))
    47  3.e3   # sqrt(md2(1,1))
    48  3.e3   # sqrt(md2(2,2))
    49  3.e3   # sqrt(md2(3,3))
"""

def softsusy(M1, M2, mu, tan_beta):
    """
    @returns SLHA output from SOFTUSY
    @rtype str
    """
    slha_str = SLHA_TEMPLATE.format(M1, M2, mu, tan_beta)
    p = subprocess.Popen([PROGRAM, "leshouches"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    return p.communicate(slha_str)[0]

def masses(M1, M2, mu, tan_beta):
    """
    @returns Absolute values of neturalino masses in ascending order, followed
    by absolute values of chargino masses in ascending order

    If SOFTSUSY fails, returns tree-level masses.
    """
    slha_str = softsusy(M1, M2, mu, tan_beta)
    try:
        slha = pyslha.readSLHA(slha_str)
    except pyslha.ParseError:
        warn("using tree-level formulas")
        return tree_level.masses(M1, M2, mu, tan_beta)
    mass = slha.blocks["MASS"]
    return np.abs(np.array([mass[k] for k in KEYS]))


if __name__ == "__main__":

    print masses(10., 100., 50., 4.)
