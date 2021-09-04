#!/usr/bin/env python3
""" Functions:
        l2_reg_create_layer(prev, n, activation, lambtha).
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Function that creates a tensorflow layer that includes
        L2 regularization
    Args:
        prev (): Is a tensor containing the output of the previous layer.
        n (int): Is the number of nodes the new layer should contain.
        activation (): Is the activation funct that should be used on the
            layer.
        lambtha (): Is the L2 regularization paramet.
    Returns:
        The output of the new layer.
    """
    raw_lay = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_regularize = tf.contrib.layers.l2_regularizer(scale=lambtha, scope=None)

    out_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_initializer=raw_lay,
                                    kernel_regularizer=new_regularize,
                                    name="layer")

    return out_tensor(prev)
