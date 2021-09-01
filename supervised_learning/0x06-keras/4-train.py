#!/usr/bin/env python3
""" Functions:
        train_model(network, data, labels, batch_size, epochs,
                    verbose=True, shuffle=False)
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ Function that  trains a model using mini-batch gradient
        descent.

    Args:
        network (Keras object): The model to train.
        data (numpy array): Numpy.ndarray of shape (m, nx)
            containing the input data.
        labels (numpy array): One-hot numpy.ndarray of shape
            (m, classes) containing the labels of data.
        batch_size (int): The size of the batch used for
            mini-batch gradient descent.
        epochs (int): The number of passes through data for
            mini-batch gradient descent.
        verbose (bool): Determines if output should be printed
            during training. Defaults to True.
        shuffle (bool): Determines whether to shuffle the batches
            every epoch. Normally, it is a good idea to shuffle,
            but for reproducibility, we have chosen to set the
            default to False. Defaults to False.
    Return:
        The History object generated after training the model.
    """
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
