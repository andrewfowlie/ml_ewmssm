"""
Bayesian analysis of MSSM.

    python mssm.py

runs two scans of the MSSM; one with log and one with linear priors. Results
should appear in `./chains/mssm*`.
"""

import numpy as np
import pymultinest

from knn_loglike import LogLike


PARAM_SAMPLES = "param_samples.npy"
LOGLIKE_SAMPLES = "loglike_samples.npy"
INDEX = "mssm.index"
LOGLIKE_SM = 4.15475279
BASENAME = "./chains/mssm_{}_"

# Make MSSM log-likelihood function

param_samples = np.load(PARAM_SAMPLES)
loglike_samples = np.load(LOGLIKE_SAMPLES)
loglike = LogLike(param_samples, loglike_samples)
loglike.lazyload(INDEX)


def wrap_loglike(cube, n_dims=4, n_params=4):
    """
    Convert cube to a numpy array, and call our log-likelihood functions.
    """
    array = np.array([cube[i] for i in range(n_dims)])
    return loglike(array) - LOGLIKE_SM

# Make MSSM priors

def linear_prior(x, min_, max_):
    """
    Map unit hypercube to a parameter with a linear prior.
    """
    return (max_ - min_) * x + min_

def log_prior(x, min_, max_, signed=False):
    """
    Map unit hypercube to a parameter with a linear prior.
    """
    if not signed:
        return min_ * (max_ / min_)**x

    sign = -1. if x < 0.5 else 1.
    x_magnitude = 2. * x - 1. if x > 0.5 else 2. * x
    return sign * min_ * (max_ / min_)**x_magnitude

def log_priors(cube, n_dims, n_params):
    """
    Map unit hypercube to M1, M2, mu and tan beta with log priors.
    """
    cube[0] = log_prior(cube[0], 1., 2000., signed=True)  # M1
    cube[1] = log_prior(cube[1], 1., 2000.)  # M2
    cube[2] = log_prior(cube[2], 1., 2000., signed=True)  # mu
    cube[3] = linear_prior(cube[3], 1., 70.)  # tan beta
    return cube

def linear_priors(cube, n_dims, n_params):
    """
    Map unit hypercube to M1, M2, mu and tan beta with linear priors.
    """
    cube[0] = linear_prior(cube[0], -2000., 2000.)  # M1
    cube[1] = linear_prior(cube[1], 1., 2000.)  # M2
    cube[2] = linear_prior(cube[2], -2000., 2000.)  # mu
    cube[3] = linear_prior(cube[3], 1., 70.)  # tan beta
    return cube

if __name__ == "__main__":

    pymultinest.run(wrap_loglike,
                    log_priors,
                    outputfiles_basename=BASENAME.format('log'),
                    n_dims=4,  # M1, M2, mu and tan beta
                    evidence_tolerance=1E-5,
                    n_live_points=2000,
                    importance_nested_sampling=False,
                    sampling_efficiency='model',
                    n_iter_before_update=100,
                    resume=True,
                    verbose=True)

    pymultinest.run(wrap_loglike,
                    linear_priors,
                    outputfiles_basename=BASENAME.format('linear'),
                    n_dims=4,  # M1, M2, mu and tan beta
                    evidence_tolerance=1E-5,
                    n_live_points=2000,
                    importance_nested_sampling=False,
                    sampling_efficiency='model',
                    n_iter_before_update=100,
                    resume=True,
                    verbose=True)
