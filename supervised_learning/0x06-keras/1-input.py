#!/usr/bin/env python3
""" Functions:
        build_model(nx, layers, activations, lambtha, keep_prob)
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """

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
    inputs = K.Input(shape=(nx,))
    regularization = K.regularizers.l2(lambtha)

    output = K.layers.Dense(layers[0], input_shape=(nx,),
                            activation=activations[0],
                            kernel_regularizer=regularization)(inputs)

    for i in range(1, len(activations)):
        drop = K.layers.Dropout(1 - keep_prob)(output)
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regularization)(drop)

    return K.Model(inputs=inputs, outputs=output)
