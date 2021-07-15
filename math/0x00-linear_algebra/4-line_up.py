#!/usr/bin/env python3
""" Linear Algebra Module
    Functions:
        add_arrays
"""


def add_arrays(arr1, arr2):
    """
        Function that adds two arrays element-wise given in
        arr1 and arr2 respectively and both should have the same shape.

        Return: New list (array) with sum result.
    """
    sum_result = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        sum_result.append(arr1[i] + arr2[i])
    return sum_result
