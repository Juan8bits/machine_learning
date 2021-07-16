#!/usr/bin/env python3
"""
    Lineal Algebra Module
    Functions:
        mat_mul
"""


def mat_mul(mat1, mat2):
    """
        Function that performs matrix multiplication
        if the two matrices cannot be multiplied return None.
        Else, return a new matrix.
    """
    if len(mat1[0]) == len(mat2):
        mult_result = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat2[0])):
                sumatory = 0
                for k in range(len(mat1[i])):
                    sumatory += mat1[i][k] * mat2[k][j]
                row.append(sumatory)
            mult_result.append(row)
        return mult_result
    return None
