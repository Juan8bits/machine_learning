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


def add_arrays(arr1, arr2):
    """
        Function that adds two arrays element-wise given in
        arr1 and arr2 respectively and both should have the same shape.

        Return: New list (array) with sum result.
    """
    sum_result = []
    if (matrix_shape(arr1) == matrix_shape(arr2)):
        shape = matrix_shape(arr1)
        for i in range(shape[0]):
            sum_result.append(arr1[i] + arr2[i])
    else:
        return None
    return sum_result
