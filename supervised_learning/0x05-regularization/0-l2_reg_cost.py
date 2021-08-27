#!/usr/bin/env python3
""" Functions:
        l2_reg_cost(cost, lambtha, weights, L, m)
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Function that calculates the cost of
        a neural network with L2 regularization.

        Cost function = Loss + (lambtha / 2 * m) *  Σ | w | ^ 2

    Args:
        cost (Numpy object): Cost of the network
            without L2 regularization.
        lambtha (float): The regularization parameter.
        weights (Numpy object): Dictionary of the
            weights and biases (numpy.ndarrays) of the
            neural network.
        L (int): The number of layers in the neural network.
        m (int): The number of data points used.
    """
    if (L == 0):
        return 0

    sum_w = 0
    for keys in weights:
        if (keys[0] == "W"):
            values = weights[keys]
            sum_w += np.linalg.norm(values)

    cost_l2 = cost + (lambtha / (2 * m)) * sum_w
    return(cost_l2)
