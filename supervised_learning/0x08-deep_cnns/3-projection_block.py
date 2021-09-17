#!/usr/bin/env python3
""" Functions:
        projection_block(A_prev, filters, s=2)
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Function that builds a projection block for resnet

    Args:
        A_prev (): is the output from previous layer.
        filters (): is tuple/list (F11, F3, F12).
            F11 first 1x1 conv.
            F3 3x3 conv.
            F12 second 1x1 conv.
    Returns: activated output of block
    """
    init = K.initializers.he_normal()
    F11, F3, F12 = filters
    conv = K.layers.Conv2D(F11, kernel_size=(1, 1),
                           strides=s,
                           padding='valid',
                           kernel_initializer=init)(A_prev)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(F3, kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           kernel_initializer=init)(relu)
    bnorm = K.layers.BatchNormalization()(conv)
    relu = K.layers.Activation('relu')(bnorm)

    conv = K.layers.Conv2D(F12, kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='valid',
                           kernel_initializer=init)(relu)
    last = K.layers.BatchNormalization()(conv)

    shortcut = K.layers.Conv2D(F12, kernel_size=(1, 1),
                               strides=s,
                               padding='valid',
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)

    relu = K.layers.Add()([last, shortcut])
    out = K.layers.Activation('relu')(relu)

    return out
