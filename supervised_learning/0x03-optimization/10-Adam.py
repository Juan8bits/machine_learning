#!/usr/bin/env python3
""" Functions:
        create_Adam_op(loss, alpha, beta1, beta2, epsilon).
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ Function that creates the training operation for
        a neural network in tensorflow using the Adam
        optimization algorithm.

    Args:
        loss (): is the loss of the network.
        alpha (): is the learning rate.
        beta1 (): is the weight used for the first moment.
        beta2 (): is the momentum weight.
        epsilon (): is a small number to avoid division by zero.

    Returns:
        The Adam optimization operation.
    """
    train = tf.train.AdamOptimizer(learning_rate=alpha,
                                   beta1=beta1,
                                   beta2=beta2,
                                   epsilon=epsilon)
    return train.minimize(loss)
