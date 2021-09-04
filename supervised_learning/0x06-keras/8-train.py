#!/usr/bin/env python3
""" Functions:
        train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False).
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
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
        learning_rate_decay (bool): A boolean that indicates whether
            learning rate decay should be used.
            Learning rate decay should only be performed if
            validation_data exists.
            The decay should be performed using inverse time decay.
            the learning rate should decay in a stepwise fashion
            after each epoch.
            Each time the learning rate updates, Keras should print
            a message.
        alpha (float): The initial learning rate.
        decay_rate (): The decay rate.
        save_best (bool): bollean indicating whether to save the model
            after each epoch if it is the best
            A model is considered the best if its validation loss is
            the lowest that the model has obtained
        filepath (str): The file path where the model should be saved
    Return:
        The History object generated after training the model.
    """
    def learning_rate(epochs):
        """ Function that updates the learning rate using
            inverse time decay.
        """
        return alpha / (1 + decay_rate * epochs)

    callb = []
    if save_best:
        mcp_save = K.callbacks.ModelCheckpoint(filepath,
                                               save_best_only=True,
                                               monitor='val_loss',
                                               mode='min')
        callb.append(mcp_save)

    if validation_data and learning_rate_decay:
        lr_decay = K.callbacks.LearningRateScheduler(learning_rate,
                                                     verbose=1)
        callb.append(lr_decay)
    if early_stopping and validation_data:
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callb.append(early_stopping)
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callb)
