#!/usr/bin/env python3
"""
    Functions:
        def kmeans(X, k, iterations=1000)
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """ Performs k-means on dataset
        Args:
            X: is a numpy.ndarray of shape (n, d) containing the dataset
                * n: is the number of data points
                * d: is the number of dimensions for each data point
            k: is a positive integer containing the number of clusters
            iterations: is a positive integer containing the maximum number
                of iterations that should be performed
            If no change in the cluster centroids occurs between iterations,
                your function should return
            Initialize the cluster centroids using a multivariate uniform
                distribution (based on 0-initialize.py)
            If a cluster contains no data points during the update step,
                reinitialize its centroid
            You should use numpy.random.uniform exactly twice
            You may use at most 2 loops
            Returns:
                * C, clss, or None, None on failure
                * C is a numpy.ndarray of shape (k, d) containing the
                    centroid means for each cluster
                * clss is a numpy.ndarray of shape (n,) containing the
                    index of the cluster in C that each data point belongs to
    """
    if type(X) is not np.ndarray or X.ndim != 2\
            or type(k) is not int or k <= 0\
            or type(iterations) is not int\
            or iterations <= 0:
        return None, None
    n, d = X.shape
    initialize = np.random.uniform(low=X.min(axis=0),
                                   high=X.max(axis=0),
                                   size=(k, d))

    # randint
    for i in range(iterations):

        distances = np.linalg.norm(X - np.expand_dims(initialize, 1), axis=2)
        clss = np.argmin(distances, axis=0)
        cent_current = initialize.copy()
        for c in range(k):
            if len(X[c == clss]) == 0:
                initialize[c] = np.random.uniform(low=X.min(axis=0),
                                                  high=X.max(axis=0),
                                                  size=(1, d))
            else:
                initialize[c] = np.mean(X[c == clss], axis=0)
        if np.all(cent_current == initialize):
            break
    distances = np.linalg.norm(X - np.expand_dims(initialize, 1), axis=2)
    clss = np.argmin(distances, axis=0)
    return initialize, clss
