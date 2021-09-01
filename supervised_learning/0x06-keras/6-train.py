#!/usr/bin/env python3
""" Functions:
        def train_model(network, data, labels, batch_size, epochs,
                        validation_data=None, early_stopping=False,
                        patience=0, verbose=True, shuffle=False).
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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
        validation_data (): The data to validate the model with.
            Defaults to None.
        verbose (bool): Determines if output should be printed
            during training. Defaults to True.
        shuffle (bool): Determines whether to shuffle the batches
            every epoch. Normally, it is a good idea to shuffle,
            but for reproducibility, we have chosen to set the
            default to False. Defaults to False.
        early_stopping (bool): A boolean that indicates whether.
            early stopping should be used.
            early stopping should only be performed if validation_data
            exists.
            early stopping should be based on validation loss.
        patience (int): The patience used for early stopping.
    Return:
        The History object generated after training the model.
    """
    callb = []
    if (validation_data):
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callb.append(early_stopping)
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callb)
