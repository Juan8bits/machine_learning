#!/usr/bin/env python3
""" Functions:
    l2_reg_cost(cost)
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """ Function that calculates the cost of a neural network
        with L2 regularization.

    Args:
        cost (td object): Tensor containing the cost of the
            network without L2 regularization.
    Return:
        A tensor containing the cost of the network accounting
        for L2 regularization.
    """
    l2_cost = tf.losses.get_regularization_losses(scope=None)
    return cost + l2_cost
