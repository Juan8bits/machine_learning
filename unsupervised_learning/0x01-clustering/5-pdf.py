#!/usr/bin/env python3
"""
    Functions:
        def pdf(X, m, S)
"""
import numpy as np


def pdf(X, m, S):
    """ probability density function of a Gaussian distribution
        Args:
            X: ndarray (n, d) dataset
            m: ndarray (d,) mean of distribution
            S: ndarray (d, d) covariance of distribution
        Returns:
            - P, or None on failure
            - P ndarray (n,) containing the PDF values for each data point
            - All values in P should have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or X.ndim != 2\
            or type(m) is not np.ndarray or m.ndim != 1\
            or type(S) is not np.ndarray or S.ndim != 2:
        return None
    if (type(X) is not np.ndarray or len(X.shape) != 2):
        return None

    n, d = X.shape

    if (type(m) is not np.ndarray or m.shape != (d,)):
        return None
    if (type(S) is not np.ndarray or S.shape != (d, d)):
        return None

    cov_det = np.linalg.det(S)
    const = 1 / (((2 * np.pi) ** (d / 2)) * (cov_det ** (1 / 2)))
    potence = ((X - m) @ np.linalg.inv(S) * (X - m)).sum(axis=1) * (- 1 / 2)
    P = const * np.exp(potence)
    P = np.where(P < 1e-300, 1e-300, P)

    return P
