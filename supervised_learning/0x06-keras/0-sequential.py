#!/usr/bin/env python3
""" Functions:
        build_model(nx, layers, activations, lambtha, keep_prob)
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Function that builds a neural network with the Keras library.

    Args:
        nx (int): The number of input features to the network.
        layers (List): List containing the number of nodes in each
            layer of the network.
        activations (List): List containing the activation functions
            used for each layer of the network.
        lambtha (Float): The L2 regularization parameter.
        keep_prob (Float): The probability that a node will be kept
            for dropout.
    """
    model = K.Sequential()
    regularization = K.regularizers.l2(lambtha)

    for i in range(len(activations)):
        if i is 0:
            model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                                     activation=activations[0],
                                     kernel_regularizer=regularization))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=regularization))
        if i < len(activations) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
