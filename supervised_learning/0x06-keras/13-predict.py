#!/usr/bin/env python3
""" Functions:
        predict(network, data, verbose)
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    Args:
        network (Keras object): Is the network model to make the prediction
            with.
        data (): is the input data to make the prediction with.
        verbose (): is a boolean that determines if output should be printed
            during the prediction process.
    Returns:
        The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
