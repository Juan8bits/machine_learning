#!/usr/bin/env python3
""" Functions:
        optimize_model(network, alpha, beta1, beta2)
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Function that sets up Adam optimization for a keras model with
        categorical crossentropy loss and accuracy metrics.

    Args:
        network (Keras object): The model to optimize.
        alpha (Float): The learning rate.
        beta1 (Float): The first Adam optimization parameter.
        beta2 (Float): The second Adam optimization parameter.
    """
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
