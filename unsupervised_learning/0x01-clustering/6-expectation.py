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
    if type(X) is not np.ndarray or len(X.shape) != 2\
            or type(pi) is not np.ndarray or len(pi.shape) != 1\
            or type(m) is not np.ndarray or m.shape != (k, d)\
            or type(S) is not np.ndarray or S.shape != (k, d, d):
        return None, None
    if (not np.isclose(np.sum(pi), 1)):
        return None, None

    n, d = X.shape
    k = pi.shape[0]
    g = []
    for j in range(k):
        p = pdf(X, m[j], S[j]) * pi[j]
        g.append(p)
    prob = np.array(g)
    hood = prob.sum(axis=0)
    prob = prob / hood
    loghood = np.sum(np.log(hood))
    return prob, loghood