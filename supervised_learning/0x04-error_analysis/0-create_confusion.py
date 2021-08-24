#!/usr/bin/env python3
""" Functions:
        create_confusion_matrix(labels, logits)
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Function That creates a confusion matrix.

    Args:
        labels (numpy object): A one-hot numpy.ndarray of shape
            (m, classes) containing the correct labels for each
            data point.
            m: is the number of data points.
            classes:  is the number of classes.
        logits (numpy object): A one-hot numpy.ndarray of shape
            (m, classes) containing the predicted labels.

    Returns:
        A confusion numpy.ndarray of shape (classes, classes)
        with row indices representing the correct labels and
        column indices representing the predicted labels.
    """
    c_matrix = np.matmul(labels.T, logits)

    return c_matrix
