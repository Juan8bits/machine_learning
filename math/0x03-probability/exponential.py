#!/usr/bin/env python3
""" Class to represents an exponential distribution.
"""


class Exponential:
    """ Class to represents an exponential distribution.
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Constructor

        Args:
            data (List): List of the data to be used to estimate
                the distribution. Defaults to None.
            lambtha (float): The expected number of occurences
                in a given time frame. Defaults to 1..
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif not len(data) > 1:
            raise ValueError('data must contain multiple values')
        else:
            self.lambtha = len(data) / sum(data)

    def pdf(self, x):
        """ Mehotd that calculates the value of the PMF for
            a given number of “successes”.

        Args:
            x (int): Is the time period (Aleatory variable).

        PDF description:
            f(x) = P(X=x) = lambda * e^(-lambda*x)
        """
        if x < 0:
            return 0
        return (self.lambtha * self.e ** (-1 * (self.lambtha ** x)))
