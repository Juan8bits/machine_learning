#!/usr/bin/env python3
""" Linear Algebra Module
    Functions:
        np_cat
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Function that concatenates two matrices along specific axis.
        Return: New numpy.ndarray object
    """
    return np.concatenate((mat1, mat2), axis=axis)
