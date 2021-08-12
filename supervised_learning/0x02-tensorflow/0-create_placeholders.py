#!/usr/bin/env python3
""" Functions:
        create_placeholders(nx, classes)
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Function that returns two placeholders, x and y,
        for the neural network.

    Args:
        nx (int): The placeholder for the input data to the neural
            network.
        classes (int): The placeholder for the one-hot labels for
            the input data.
    """
    x = tf.placeholder("float", (None, nx), "x")
    y = tf.placeholder("float", (None, classes), "y")
    return x, y
