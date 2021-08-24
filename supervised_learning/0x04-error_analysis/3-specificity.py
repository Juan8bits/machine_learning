#!/usr/bin/env python3
""" Funtions:
        specificity(confusion)
"""
import numpy as np


def specificity(confusion):
    """ Function that calculates the specificity for each class
        in a confusion matrix
    Arg:
        confusion (Numpy array): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.
            classes: The number of classes
    Return:
        A numpy.ndarray of shape (classes,) containing the specificity
        of each class.
    """
    # True positive
    tp = np.diagonal(confusion)
    # False negative
    fn = np.sum(confusion, axis=1) - tp
    # False positive
    fp = np.sum(confusion, axis=0) - tp
    # True negative
    tn = np.sum(confusion) - (tp + fn + fp)
    specificity = tn / (tn + fp)

    return specificity
