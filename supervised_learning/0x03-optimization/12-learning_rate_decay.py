#!/usr/bin/env python3
""" Functions:
        learning_rate_decay(alpha, decay_rate, global_step, decay_step)
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Function that updates the learning rate using inverse time
        decay in numpy.
    Args:
        alpha (): is the original learning rate.
        decay_rate (): is the weight used to find the rate at which α will decay.
        global_step (): is the # of passes of gradient descent that have elapsed.
        decay_step (): is the number of passes of gradient descent that should
            occur before alpha is decayed further.

    Returns:
        The updated value for alpha.
    """
    inverse_train = tf.train.inverse_time_decay(learning_rate=alpha,
                                     global_step=global_step,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate,
                                     staircase=True)
    return inverse_train
