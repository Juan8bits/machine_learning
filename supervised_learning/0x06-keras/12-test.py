#!/usr/bin/env python3
""" Functions:
        test_model(network, data, labels, verbose)
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Function that tests a neural network.
    Args:
        network (Keras object): Is the network model to test.
        data (): is the input data to test the model with.
        labels (): are the correct one-hot labels of data.
        verbose (): is a boolean that determines if output should be printed
          during the testing process.
    Returns:
        The loss and accuracy of the model with the testing data,
        respectively.
    """
    return network.evaluate(data, labels, verbose=verbose)
