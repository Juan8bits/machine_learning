#!/usr/bin/env python3
""" Functions:
        create_train_op(loss, alpha).
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """ Function that creates the training operation for the network.

    Args:
        loss (tensor object): The loss of the network’s prediction.
        alpha (tensor object): the learning rate.
    Returns:
        An operation that trains the network using gradient descent.
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
