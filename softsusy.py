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
    3   {3}  # tanbeta(mZ)^DRbar
BLOCK EXTPAR     Q=  3.002744880346726e+03
    1  {0}  # bino mass parameter M1
    2  {1}  # wino mass parameter M2
    3   4.999835372666907e+03   # gluino mass parameter M3
    23  {2} # mu parameter
    26  3.000003548804418e+03   # mA pole mass
    31  3.000003548804418e+03   # sqrt(ml2(1,1))
    32  3.000003575648636e+03   # sqrt(ml2(2,2))
    33  3.000011270101646e+03   # sqrt(ml2(3,3))
    34  2.999993744299581e+03   # sqrt(me2(1,1))
    35  2.999993798271672e+03   # sqrt(me2(2,2))
    36  3.000009286843936e+03   # sqrt(me2(3,3))
    41  2.999738724081019e+03   # sqrt(mq2(1,1))
    42  2.999738731883629e+03   # sqrt(mq2(2,2))
    43  2.999779948524007e+03   # sqrt(mq2(3,3))
    44  2.999741981728408e+03   # sqrt(mu2(1,1))
    45  2.999741982627098e+03   # sqrt(mu2(2,2))
    46  2.999796825865484e+03   # sqrt(mu2(3,3))
    47  2.999735317691816e+03   # sqrt(md2(1,1))
    48  2.999735332514917e+03   # sqrt(md2(2,2))
    49  2.999763723561374e+03   # sqrt(md2(3,3))
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
