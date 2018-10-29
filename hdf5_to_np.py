"""
Convert hdf5 MSSM samples into minimal numpy files (*.npy). E.g.,

    python hdf5_to_np.py MSSMEW_all_geq500k.smallset.hdf5

This assumes, of course, particular hdf5 keys. This is neccessary to
prepare our data.
"""

import sys
import h5py
import numpy as np


def traverse_keys(dict_, lead=""):
    """
    Traverse all keys in a nested dictionary-like object.
    """
    for key, value in dict_.items():
        combine = "/".join([lead, key]).strip("/")
        if hasattr(value, "keys"):
            for subkey in traverse_keys(value, combine):
                yield subkey
        else:
            yield combine

def inspect_hdf5(file_name):
    """
    Inspect keys in hdf5 file.
    """
    with h5py.File(file_name, 'r') as hdf5:
        for key in traverse_keys(hdf5):
            print key

def hdf5_to_numpy(file_name, param_keys, loglike_key):
    """
    Make numpy arrays of relevant data from hdf5 file.
    """
    with h5py.File(file_name, 'r') as hd5:

        loglike_samples = hd5[loglike_key].value
        param_samples = np.array([hd5[p].value for p in param_keys]).T

    np.save("param_samples.npy", param_samples)
    np.save("loglike_samples.npy", loglike_samples)


if __name__ == "__main__":

    hdf5_name = sys.argv[1]

    inspect_hdf5(hdf5_name)

    loglike_key = "MSSMEW/LogLike"
    # For some files this is neccessary:
    # #MSSM11atQ_mA_parameters @MSSM11atQ_mA::primary_parameters::
    param_str = "MSSMEW/{}"
    param_names = ["M1", "M2", "mu", "TanBeta"]
    param_keys = [param_str.format(n) for n in param_names]

    hdf5_to_numpy(hdf5_name, param_keys, loglike_key)
