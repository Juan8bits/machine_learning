#!/usr/bin/env python3


def matrix_shape(matrix):
    """
        Function that calculates the shape of a matrix given in
        matrix variable.

        Return: List of integers that represent the matrix shape
    """
    shape = []
    dimention = matrix
    while (isinstance(dimention, list)):
        shape.append(len(dimention))
        dimention = dimention[0]
    return shape


def add_matrices2D(mat1, mat2):
    """
        Funtion that add two matrices element-wise given in mat1 and mat2.

        Return: New matrix with sum result.
    """
    sum_result = []
    if (matrix_shape(mat1) == matrix_shape(mat2)):
        shape = matrix_shape(mat1)
        for row in range(shape[0]):
            empty_row = []
            for column in range(shape[1]):
                empty_row.append(mat1[row][column] + mat2[row][column])
            sum_result.append(empty_row)
    else:
        return None
    return sum_result
