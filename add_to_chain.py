"""
Add quantities to the chain
===========================
"""

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def add_to_chain(func, old_name, new_name=None, n_dim=4):
    """
    @param func Function to apply to parameters in the chain
    @param old_name Name of file on disk
    @param n_dim Number of parameters in chain
    """
    old_chain = np.loadtxt(old_name, unpack=True)
    params = old_chain[2:2 + n_dim]
    add = Parallel(n_jobs=-1)(delayed(lambda x: func(*x))(x) for x in tqdm(zip(*params)))
    new_chain = np.append(old_chain, add.T, axis=0)

    default_name = "{}_modified.txt".format(old_name.replace(".txt", ""))
    new_name = default_name if new_name is None else new_name
    np.savetxt(new_name, new_chain.T)


if __name__ == "__main__":

    import sys
    from softsusy import masses

    NAME = sys.argv[1]
    add_to_chain(masses, NAME)
