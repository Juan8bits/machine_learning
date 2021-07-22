#!/usr/bin/env python3
""" Calculus module
    Functions:
        summation_i_squared(n)
"""


def summation_i_squared(n):
    """ Write a function that calculates a summation
        where n is the stopping condition.

        Parameters:
            n - Stopping summation condition
        Return:
            Integer value of the sum, if n is not a valid
            number return None.
    """
    if isinstance(n, int) is False:
        return None

    initial_condition = 1
    summatory = 0
    for i in range(initial_condition, n + 1):
        summatory += i**2
    return summatory
