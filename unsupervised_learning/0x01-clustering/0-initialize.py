#!/usr/bin/env python3
"""
    Functions:
        def initialize(X, k)
"""
import numpy as np


def initialize(X, k):
    """ Initialized cluster centroids for K-means

        Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset that will
            be used for K-means clustering
            * n: is the number of data points
            * d: is the number of dimensions for each data point
        k: is a positive integer containing the number of clusters
        The cluster centroids should be initialized with a multivariate uniform
            distribution along each dimension in d:
            * The minimum values for the distribution should be the minimum
                values of X along each dimension in d
            * The maximum values for the distribution should be the maximum
                values of X along each dimension in d
    """
    if type(X) is not np.ndarray or X.ndim != 2\
            or type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    initialize = np.random.uniform(low=X.min(axis=0),
                                   high=X.max(axis=0),
                                   size=(k, d))
    return initialize
