#!/usr/bin/env python3
import numpy as np
""" Linear Algebra Module
    Functions:
        np_elementwise
"""


def np_elementwise(mat1, mat2):
    """ Function that performs element-wise addition, subtraction,
        multiplication, and division.
        Return: Tuple that containing the element-wise sum,
            difference, product, and quotient, respectively.
    """
    operations_result = (np.add(mat1, mat2),
                         np.subtract(mat1, mat2),
                         np.multiply(mat1, mat2),
                         np.divide(mat1, mat2))
    return operations_result
