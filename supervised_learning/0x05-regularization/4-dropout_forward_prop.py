#!/usr/bin/env python3
""" Functions:
        dropout_forward_prop(X, weights, L, keep_prob)
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Function that conducts forward propagation using Dropout.
    Args:
        X (Numpy array): Numpy.ndarray of shape (nx, m) containing the
            input data for the network.
            nx is the number of input features.
            m is number of data points.
        weights (dict): dictionary of the weights and biases of the neural
            network.
        L (int): Number of layers in the network.
        keep_prob (float): Probability that a node will be kept.
    Return:
        A dictionary containing the outputs of each layer and the dropout mask
        used on each layer (see example for format).
    """
    cache = {}
    cache['A0'] = X
    for layer in range(1, L + 1):
        current_W = weights['W' + str(layer)]
        current_b = weights['b' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        z = (np.matmul(current_W, A_prev)) + current_b
        dropout = np.random.binomial(1, keep_prob, size=z.shape)

        if layer is L:
            t = np.exp(z)
            cache['A' + str(layer)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache['A' + str(layer)] = np.tanh(z)
            cache['D' + str(layer)] = dropout
            cache['A' + str(layer)] *= dropout
            cache['A' + str(layer)] /= keep_prob
    return (cache)
