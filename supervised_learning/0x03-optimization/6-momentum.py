#!/usr/bin/env python3
""" Functions:
        create_momentum_op(loss, alpha, beta1)
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Function that creates the training operation for a
        neural network in tensorflow using the gradient
        descent with momentum optimization algorithm.

    Args:
        loss (): is the loss of the network.
        alpha (): is the learning rate.
        beta1 (): is the momentum weight.

    Returns:
        The momentum optimization operation.
    """
    train = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)

    op = train.minimize(loss)
    return op
