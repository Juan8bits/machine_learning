#!/usr/bin/env python3
""" One Hot encode Function
"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Function that converts a numeric label
        vector into a one-hot matrix.

    Args:
        Y (numpy object): Numpy.ndarray with shape (m,)
            containing numeric class labels m is the number
            of examples.
        classes (int): The maximum number of classes found in Y.

    Return:
        One-hot encoding of Y with shape (classes, m),
        or None on failure.
    """
    if not isinstance(Y, np.ndarray) or not Y:
        return None
    if not isinstance(classes, int) or classes <= np.amax(Y):
        return None
    output = np.zeros((Y.size, Y.max()+1))
    output[np.arange(Y.size), Y] = 1
    return output.T
