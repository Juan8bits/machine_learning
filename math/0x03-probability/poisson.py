#!/usr/bin/env python3
""" Class to represent a poisson distribution
"""


class Poisson:
    """ Class that represent a poisson distribution
    """
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Constructor

        Args:
            data (list): Is a list of the data to be used to estimate
                the distribution. Defaults to None.
            lambtha (float): The expected number of occurences in a
                given time frame. Defaults to 1..
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
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """ Mehotd that calculates the value of the PMF for
            a given number of “successes”.

        Args:
            k (int): Is the number of “successes” (Aleatory variable).

        PMF description:
            f(x) = P(X=x) = ((e^-lambda) * (lambda^ x)) / x!
        """
        k = int(k)
        if k < 0:
            return 0
        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i
        return ((self.e ** (-1 * self.lambtha)) * (self.lambtha ** k)) / k_fact

    def cdf(self, k):
        """ Method that calculates the value of the CDF for a
            given number of “successes”

        Args:
            k (int): Is the number of “successes” (Aleatory variable).

        CDF description:
            P(X=0) + P(X>=1) = 1
         """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
