#!/usr/bin/env python3
""" Functions:
        sensitivity(confusion)
"""
import numpy as np


def precision(confusion):
    """ Function that calculates the precision for each class in a
        confusion matrix,
    Args:
        confusion (Numpy array): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.
            classes: The number of classes.
    Return:
        A numpy.ndarray of shape (classes,) containing the precision
        of each class.
    """
    true_positive = np.diagonal(confusion)
    all_positive = np.sum(confusion, axis=0)
    precision = true_positive / all_positive

    return precision
