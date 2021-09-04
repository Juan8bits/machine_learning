#!/usr/bin/env python3
""" Functions:
        dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L).
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Function that creates a tensorflow layer that includes L2
        regularization
    Args:
        prev (Numpy array): Is a tensor containing the output of the
            previous layer.
        n (int): Is the number of nodes the new layer should contain.
        activation (str): Is the activation funct that should be used
            on the layer.
        lambtha (): Is the L2 regularization paramet.
    Returns:
        The output of the new layer.
    """
    weights2 = weights.copy()
    m = Y.shape[1]

    for neural_lyr in reversed(range(L)):
        n = neural_lyr + 1
        if (n == L):
            dz = cache["A" + str(n)] - Y
            dw = (np.matmul(cache["A" + str(neural_lyr)], dz.T) / m).T
        else:
            dz1 = np.matmul(weights2["W" + str(n + 1)].T, current_dz)
            dz2 = 1 - cache["A" + str(n)]**2
            dz = dz1 * dz2 * cache['D' + str(n)] / keep_prob
            dw = np.matmul(dz, cache["A" + str(neural_lyr)].T) / m

        db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W" + str(n)] = weights["W" + str(n)] - (alpha * dw)
        weights["b" + str(n)] = weights["b" + str(n)] - alpha * db
        current_dz = dz
