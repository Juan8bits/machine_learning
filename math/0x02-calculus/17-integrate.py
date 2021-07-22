#!/usr/bin/python3
""" Calculus Module
    Functions:
        poly_integral(poly, C=0)
"""


def poly_integral(poly, C=0):
    """ Function that calculates the integral of a plynominal

    Args:
        poly ([type]): [description]
        C (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    if isinstance(poly, list) is False or len(poly) is 0 \
       or isinstance(C, int) is False:
        return None
    if len(poly) == 1:
        return poly
    integral = [C, poly[0]]
    for i in range(1, len(poly)):
        if poly[i] == 0:
            integral.append(0)
        else:
            integral.append(poly[i]/(i + 1))
    return integral
