#!/usr/bin/env python3
"""
    Linear Algebra Module
    Functions:
        deepcopy
        cat_matrices2D
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Function that concatenates two 2D matrices along specific axis
        Return: New matrix
    """
    # make a deep copy
    conc_result = [row[:] for row in mat1]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for row in mat2:
            conc_result.append(row)
    elif axis == 1 and len(mat1) == len(mat2):
        for col in range(len(mat2)):
            conc_result[col] += mat2[col]
    else:
        return None
    return conc_result
