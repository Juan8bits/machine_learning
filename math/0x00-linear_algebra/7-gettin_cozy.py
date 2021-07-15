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
    new_matrix = [row[:] for row in matrix]
    return new_matrix


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
            conc_result += mat2
        elif (i == axis and axis == 1):
            for j in range(len(conc_result)):
                conc_result[j] += list(mat2[j])
        pass
    if (conc_result == mat1):
        return None
    return conc_result
