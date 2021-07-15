#!/usr/bin/env python3
""" Linear Algebra module. Function matrix_shape """


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
