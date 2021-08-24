#!/usr/bin/env python3
""" Functions:
        sensitivity(confusion)
"""
import numpy as np


def sensitivity(confusion):
    """ Function that calculates the sensitivity for each class in a
        confusion matrix.
    Args:
        confusion (Numpy object): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.
            classes: The number of classes.
    Return:
        A numpy.ndarray of shape (classes,) containing the sensitivity
        of each class.
    """
    positive = np.sum(confusion, axis=1)
    true_positive = np.diagonal(confusion)
    sensitivity = true_positive / positive

    return np.array(sensitivity)
