#!/usr/bin/env python3
"""
    Linear Algebra Module
    Functions:
        deepcopy
        cat_matrices2D
"""


def deepcopy(matrix):
    """
        Function that make a deep copy to matrix
        Return: New matrix
    """
    return [row[:] for row in matrix]


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Function that concatenates two 2D matrices along specific axis
        Return: New matrix
    """
    if len(mat1) == 0 and len(mat2) == 0 or\
            len(mat1[0]) == 0 and len(mat2[0]) == 0:
        return None
    conc_result = deepcopy(mat1)
    for i in range(0, 2):
        if i == axis and axis == 0:
            for row in mat2:
                conc_result.append(row)
        elif (i == axis and axis == 1):
            for col in range(len(conc_result)):
                conc_result[col] += list(mat2[col])
        pass
    if (conc_result == mat1):
        return None
    return conc_result
