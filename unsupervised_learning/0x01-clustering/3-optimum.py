#!/usr/bin/env python3
"""
    Functions:
        def optimum_k(X, kmin=1, kmax=None, iterations=1000)   
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance
        Args:
            X: ndarray (n, d) dataset to cluster
            kmin: positive int min # of clusters to check for (inclusive)
            kmax: positive int max # of clusters to check for (inclusive)
            iterations: positive int
            This function should analyze at least 2 different cluster sizes
        Returns:
            - results, d_vars, or None, None on failure
            - results list outputs of K-means for each cluster size
            - d_vars list difference in variance from the
                smallest cluster size for each cluster size
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    n, d = X.shape
    if kmax is None:
        kmax = n
    if type(kmin) is not int or kmin < 1\
            or type(kmax) is not int or kmax < 1\
            or type(iterations) is not int or iterations < 1\
            or kmin >= kmax:
        return None, None
    results = []
    d_vars = []
    for k in range(kmin, kmax + 1):
        centroids, clss = kmeans(X, k, iterations=iterations)
        var = variance(X, centroids)
        if k == kmin:
            mini = var
        results.append((centroids, clss))
        d_vars.append(mini - var)
    return results, d_vars
