#!/usr/bin/env python3
""" Linear Algebra Module
    Functions:
        no_slice
"""
import numpy as np


def np_slice(matrix, axes={}):
    """ Function that slices a matrix along specific axes.

        Parameters:
            axes: Is a dictionary where the key is an axis
                to slice along and the value is a tuple representing
                the slice to make along that axis.
            matrix: numpy.ndarray.
        Return:
            New numpy.ndarray sliced
    """
    slices = []
    for axs in range(len(matrix)):
        if axs in axes.keys():
            value = axes.get(axs)
            slices.append(slice(*value))
        else:
            slices.append(slice(None))
    return np.array(matrix[tuple(slices)])
