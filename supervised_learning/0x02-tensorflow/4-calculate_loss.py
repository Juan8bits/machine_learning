#!/usr/bin/env python3
""" Functions:
        calculate_loss(y, y_pred)
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Function that calculates the softmax cross-entropy
        loss of a prediction.

    Args:
        y (tensor object): Placeholder for the labels of the input data.
        y_pred (tensor object): Tensor containing the network’s predictions.
    Return:
        A tensor containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
