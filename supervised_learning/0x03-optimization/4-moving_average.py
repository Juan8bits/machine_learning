#!/usr/bin/env python3
""" Functions:
        moving_average(data, beta).
"""
import numpy as np


def moving_average(data, beta):
    """ Funtions that calculates the weighted moving average
        of a data set.
    Args:
        data (): is the list of data to calculate the moving average of
        beta (): is the weight used for the moving average.
    Return:
        A list containing the moving averages of data.
    """
    averages = []
    mov_avg = 0
    for i in range(len(data)):
        mov_avg = ((mov_avg * beta) + ((1 - beta) * data[i]))
        correction_bias = 1 - (beta ** (i + 1))
        averages.append(mov_avg / correction_bias)
    return averages
