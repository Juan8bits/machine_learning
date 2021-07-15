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
    if (len(mat1) == 0 and len(mat2) == 0) or\
            (len(mat1[0]) == 0 and len(mat2[0]) == 0):
        return None
    # make a deep copy
    conc_result = [row[:] for row in mat1]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for row in mat2:
            conc_result.append(row)
<<<<<<< HEAD
    elif axis == 1 and len(mat1) == len(mat2):
        for col in range(len(mat2)):
            conc_result[col] += mat2[col]
=======
    elif axis == 1:
        for col in range(len(conc_result)):
            conc_result[col] += list(mat2[col])
>>>>>>> c711f4e34743b00c8141520bebc922a1b63f15e1
    else:
        return None
    return conc_result
