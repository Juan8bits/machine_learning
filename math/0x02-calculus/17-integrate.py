#!/usr/bin/env python3
""" Calculus Module
    Functions:
        poly_integral(poly, C=0)
"""


def poly_integral(poly, C=0):
    """ Function that calculates the integral of a plynominal

    Args:
        poly (list): List of coefficients representing a polynomial
        C (int, optional): Integer representing the integration constant.
                            Defaults to 0.
    Returns:
        New list of coefficients representing the integral of
        the polynomial. If poly or C are not valid, return None.
    """
    if isinstance(poly, list) is False or len(poly) is 0 \
       or isinstance(C, int) is False:
        return None
    if len(poly) == 1:
        return [C, poly[0]]
    integral = [C, poly[0]]
    for i in range(1, len(poly)):
        if poly[i] % (i + 1) is 0:
            integral.append(poly[i]//(i + 1))
        else:
            integral.append(poly[i]/(i + 1))
    return integral
