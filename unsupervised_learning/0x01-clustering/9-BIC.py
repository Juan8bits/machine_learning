#!/usr/bin/env python3
"""
    Functions:
        def BIC(X, kmin=1, kmax=None, iterations=1000,
            tol=1e-5, verbose=False)
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ finds the best number of clusters for a GMM using the Bayesian
    Information Criterion (BIC)
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the data set
        kmin: is a positive integer containing the minimum number of
            clusters to check for (inclusive)
        kmax: is a positive integer containing the maximum number of
            clusters to check for (inclusive)
        iterations: is a positive integer containing the maximum
            number of iterations for K-means
        tol: is a non-negative float containing tolerance of the log
            likelihood verbose is a boolean that determines if you
            should print information
    Returns:
        - best_k, best_result, l, b, or None, None, None, None on failure
        - best_k is the best value for k based on its BIC
        - best_result is tuple containing pi, m, S
            - pi is a numpy.ndarray of shape (k,) containing the cluster priors
                for the best number of clusters
            - m is a numpy.ndarray of shape (k, d) containing the centroid
                means for the best number of clusters
            - S is a numpy.ndarray of shape (k, d, d) containing the covariance
                matrices for the best number of clusters
            - l is a numpy.ndarray of shape (kmax - kmin + 1) containing the
                log likelihood for each cluster size tested
            - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
                BIC value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int):
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    if kmax is None:
        kmax = iterations

    n = X.shape[0]
    prior_bic = 0
    likelyhoods = bics = []
    best_k = kmax
    pi_prev = m_prev = S_prev = best_res = None
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol,
                                                   verbose)
        bic = k * np.log(n) - 2 * ll
        if np.isclose(bic, prior_bic) and best_k >= k:
            best_k = k - 1
            best_res = pi_prev, m_prev, S_prev
        pi_prev, m_prev, S_prev = pi, m, S
        likelyhoods.append(ll)
        bics.append(bic)
        prior_bic = bic

    return best_k, best_res, np.asarray(likelyhoods), np.asarray(bics)
