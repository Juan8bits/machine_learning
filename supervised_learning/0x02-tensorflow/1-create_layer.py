#!/usr/bin/env python3
""" Functions:
        create_layer(prev, n, activation)
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Function that create a layer.

    Args:
        prev (tensor object): The tensor output of the previous layer,
        n (int): The number of nodes in the layer to create.
        activation (tensor object): The activation function that the
            layer should use.
    """
    w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    out = tf.layers.dense(inputs=prev, units=n, activation=activation,
                          name="layer", kernel_initializer=w)
    return out
