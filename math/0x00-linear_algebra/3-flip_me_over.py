#!/usr/bin/env python3


def create_empty_transpose_2D_matrix(shape):
    """
        Function that creates an empty 2D matrix
        in list python object format.

        Return: Return an empty transpose matrix with "shape"
                dimentions
    """
    matrix = []
    for i in range(shape[1]):
        sub = []
        for j in range(shape[0]):
            sub.append(0)
        matrix.append(sub)
    return matrix


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


def matrix_transpose(matrix):
    """
        Given a 2D matrix, return the transpose of a that matrix.

        Return: A new matrix wich is the transpose of a given matrix
    """
    size = matrix_shape(matrix)
    transpose_matrix = create_empty_transpose_2D_matrix(size)
    # Fill the transpose matrix
    for row in range(len(matrix)):
        for column in range(len(matrix[0])):
            transpose_matrix[column][row] = matrix[row][column]
    return transpose_matrix


"""
# Easy way
def matrix_transpose(matrix):
    import numpy as np
    np_matrix = np.array(matrix)
    return (np_matrix.transpose())
"""
