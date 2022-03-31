#!/usr/bin/env python3
"""
    Functions:
        def variance(X, C)
"""
import numpy as np


def variance(X, C):
    """ Calculates the total intra-cluster variance
        Args:
            X: ndarray (n, d) dataset to cluster
            C: ndarray (k, d) centroid means for each cluster
        Returns: var, or None on failure
            - var is the total variance
    """
    if type(X) is not np.ndarray or X.ndim != 2\
            or type(C) is not np.ndarray or C.ndim != 2:
        return None
    try:
        k, d = C.shape
        '''n, _ = X.shape
        k, d = C.shape'''
        dist = np.linalg.norm(X - np.expand_dims(C, 1), axis=-1)
        min = np.min(dist, axis=0)
        return np.sum(np.square(min))
    except Exception:
        return None
