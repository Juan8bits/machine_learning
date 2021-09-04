#!/usr/bin/env python3
""" Functions:
        save_weights(network, filename, saveformat).
        load_weights(network, filename).
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Function that saves a model’s weights.
    Args:
        network (Keras object): is the model whose weights should be saved.
        filename (str): Is the path of the file that the weights should be
            saved to.
        save_format (str): Is the format in which the weights should be saved.
    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads a model’s weights
    Args:
        network (Keras object): Is the model to which the weights should
            be loaded.
        filename (str): Is the path of the file that the weights should
            be loaded from.
    Returns:
        None
    """
    network.load_weights(filename)
    return None
