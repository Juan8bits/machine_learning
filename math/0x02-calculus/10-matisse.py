#!/usr/bin/env python3
"""Calculus module
"""


def poly_derivative(poly):
    """ Function that

    Args:
        poly (List): List of coefficients representing a polynomial.
                    The index of the lst represents the power of x
                    that the coeffucient belongs
    Return:
        New list of coefficients representing the derivate of the polynomial.
        if poly is not valid return None
    """
    if isinstance(poly, list) is False:
        return None
    if len(poly) == 1:
        return [0]
    derivate = []
    for i in range(1, len(poly)):
        if poly[i] == 0:
            derivate.append(0)
        elif i == 1:
            derivate.append(poly[i])
        else:
            derivate.append(i * poly[i])
    return derivate
