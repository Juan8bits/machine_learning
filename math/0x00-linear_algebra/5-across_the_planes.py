#!/usr/bin/env python3
""" Linear Algebra module
    Functions:
        add_matrices2D
"""


def add_matrices2D(mat1, mat2):
    """
        Funtion that add two matrices element-wise given in mat1 and mat2.

        Return: New matrix with sum result.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    sum_result = []
    for row in range(len(mat1)):
        empty_row = []
        for column in range(len(mat1[0])):
            empty_row.append(mat1[row][column] + mat2[row][column])
        sum_result.append(empty_row)
    return sum_result
