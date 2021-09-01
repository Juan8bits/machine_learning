#!/usr/bin/env python3
""" Functions:
        one_hot(labels, classes=None)
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Function that converts a label vector into a one-hot matrix.

    Args:
        labels (List): Input labels.
        classes (): The last dimension of the one-hot matrix.
            Defaults to None.
    """
    one_hot_matrix = K.utils.to_categorical(labels, classes)

    return one_hot_matrix
