#!/usr/bin/env python3
"""
    Functions:
        def expectation(X, pi, m, S)
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ expectation step in the EM algorithm for a GMM
        Args:
            X: ndarray (n, d) dataset
            pi: ndarray (k,) priors for each cluster
            m: ndarray (k, d) centroid means
            S: ndarray (k, d, d) covariance of distribution
        Returns:
            - g, l, or None, None on failure
            - g ndarray (k, n) containing posterior probabilities
            - l total log likelihood
    """
    if (type(X) is not np.ndarray or len(X.shape) != 2):
        return None, None
    if (type(pi) is not np.ndarray or len(pi.shape) != 1):
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if (type(m) is not np.ndarray or m.shape != (k, d)):
        return None, None
    if (type(S) is not np.ndarray or S.shape != (k, d, d)):
        return None, None
    if (not np.isclose(np.sum(pi), 1)):
        return None, None

    likelihood = []

    for i in range(k):
        likelihood.append(pdf(X, m[i], S[i]))

    likelihood = np.array(likelihood)
    l_pi = likelihood * pi[:, np.newaxis]
    marginal_p = np.sum(likelihood * pi[:, np.newaxis], axis=0)
    g = l_pi / marginal_p[:, np.newaxis].T
    tll = (np.log(l_pi.sum(axis=0))).sum()  # l

    return g, tll
