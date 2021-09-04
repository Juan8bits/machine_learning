#!/usr/bin/env python3
""" Functions:
        save_model(network, filename)
        load_model(filename)
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ Function that saves an entire model.
    Args:
        network (Keras model): Is the model to save.
        filename (str): Is the path of the file that the model
            should be saved to.
    Returns:
        None.
    """
    network.save(filename)
    return None


def load_model(filename):
    """ Function that loads an entire model.

    Args:
        filename (str): Is the path of the file that the model
            should be loaded from.
    Returns:
        The loaded model.
    """
    network = K.models.load_model(filename)
    return network
