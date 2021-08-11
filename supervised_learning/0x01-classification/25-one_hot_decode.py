#!/usr/bin/env python3
"""One Hot Decode Function
"""
import numpy as np


def one_hot_decode(one_hot):
    """ Function that converts a one-hot matrix
        into a vector of labels.
    Arg:
        one_hot(numpy object): One-hot encoded numpy.ndarray
            with shape (classes, m).
            classes: is the maximum number of classes.
            m: is the number of examples.
    Return:
        A numpy.ndarray with shape (m, ) containing the numeric
        labels for each example, or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) is not 2:
        return None
    decode = np.argmax(one_hot, axis=0)
    return decode
