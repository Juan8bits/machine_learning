#!/usr/bin/env python3
""" Functions:
        dropout_create_layer(prev, n, activation, keep_prob)
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ Funtion that creates a layer of a neural network using dropout.

    Args:
        prev (): Tensor containing the output of the previous layer.
        n (int): Number of nodes the new layer should contain.
        activatio (str): Activation funct that should be used on the layer.
        keep_prob (): The probability that a node will be kept.
    Returns:
        The output of the new layer.
    """
    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout_layer = tf.layers.Dropout(keep_prob)

    output_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_regularizer=dropout_layer,
                                    kernel_initializer=raw_layer)
    return output_tensor(prev)
