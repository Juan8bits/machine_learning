#!/usr/bin/env python3
""" Functions:
        normalization_constants(X)
"""
import numpy as np


def normalize(X, m, s):
    """ Function that normalizes (standardizes) a matrix.

    Args:
        X (numpy array): Numpy.ndarray of shape (d, nx) to normalize.
            d is the number of data points.
            nx is the number of features.
        m (numpy array): Numpy.ndarray of shape (nx,) that contains the
            mean of all features of X.
        s (numpy array): Numpy.ndarray of shape (nx,) that contains the
            standard deviation of all features of X.
    Return:
        The normalized X matrix.
    """
    X = (X - m) / s
    return X
