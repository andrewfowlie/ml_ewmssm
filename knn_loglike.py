"""
Approximate log-likelihood function from samples using approximate kNN
regression.

We define a class `LogLike` that takes samples of log-likelihood and the
parameters.

    python knn_loglike.py

Runs and times the code on a simple case.
"""

from time import time

import numpy as np
import mrpt


class LogLike(mrpt.MRPTIndex):
    """
    Estimate log-likelihood from a set of samples using
    approximate k nearest neighbours
    """
    def __init__(self, param_samples, loglike_samples, depth=10, n_trees=20, **kwargs):
        """
        Build an approximate k nearest neigbours object
        @param param_samples Parameters for samples
        @type param_samples np.array
        @param loglike_samples Log-likeliood corresponding to parameter samples
        @type loglike_samples np.array
        """
        if not loglike_samples.shape[0] == param_samples.shape[0]:
            raise ValueError("number of loglike and param samples do not match")

        self.loglike_samples = loglike_samples
        self.like_samples = np.exp(loglike_samples)

        min_ = np.amin(param_samples, axis=0)
        range_ = np.amax(param_samples, axis=0) - min_
        self.map_to_unit = lambda params: (params - min_) / range_

        unit_samples = self.map_to_unit(param_samples)
        self.unit_samples = np.ascontiguousarray(unit_samples, dtype="float32")
        super(LogLike, self).__init__(self.unit_samples, depth, n_trees, **kwargs)

    def __call__(self, params, k=None, weighted=True, average_loglike=False, **kwargs):
        """
        @param params Parameter point at which to estimate log-likelihood
        @param k How many nearest neighbours to use
        @param weighted Whether to weight the average from nearest neighbours
        @param average_loglike Whether to average log-likelihood or likelihood
        @returns Estimate of log-likelihood from nearest neighbours
        """
        unit = self.map_to_unit(params).astype('float32')
        k = len(params) + 1 if k is None else k
        indices, distances = self.ann(unit, k, return_distances=True, **kwargs)
        likes = self.like_samples[indices]
        loglikes = self.loglike_samples[indices]

        exact = likes[distances == 0.]
        if len(exact) >= 1:
            return np.log(exact[0])

        if weighted:
            weights = np.array(1. / distances)
            weights /= weights.sum()
            return np.dot(loglikes, weights) if average_loglike else np.log(np.dot(likes, weights))

        return np.mean(loglikes) if average_loglike else np.log(np.mean(likes))

    def lazyload(self, file_name):
        """
        Try to load an index file. If we can't, build one and save it.

        @param file_name Name of index file
        """
        try:
            print "Loading index = {}".format(file_name)
            self.load(file_name)
        except IOError:
            print "Index doesn't exist -  building it from scratch. May take a few minutes."
            t = time()
            self.build()
            t = time() - t
            print "Index built in {} seconds".format(t)
            self.save(file_name)
            print "Saved index = {}".format(file_name)

if __name__ == "__main__":

    # Generate synthetic test data similar to MSSM data
    N_SAMPLES = 2440105
    N_DIM = 4
    INDEX = 'example.index'
    param_samples = np.random.rand(N_SAMPLES, N_DIM)
    loglike_samples = np.random.rand(N_SAMPLES)

    # Construct our log-likelihood object and build the index
    loglike = LogLike(param_samples, loglike_samples)
    loglike.lazyload(INDEX)

    # Time the performance
    query = np.random.rand(N_DIM)
    t = time()
    l = loglike(query)
    t = time() - t
    print 'Calculated loglike from {} in {} seconds'.format(INDEX, t)
