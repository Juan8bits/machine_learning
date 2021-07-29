#!/usr/bin/env python3
""" Class to represents a Binomial distribution.
"""


class Binomial:
    """ Class to represents a Binomial distribution.
    """
    def __init__(self, data=None, n=1, p=0.5):
        """ Constructor

        Args:
            data (List): List of the data to be used to estimate
                the distribution. Defaults to None.
            n (int): The number of Bernoulli trials. Defaults to 1.
            p (float):  The probability of a “success”. Defaults to 0.5.
        """
        if data is None:
            if n < 1:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif not len(data) > 1:
            raise ValueError('data must contain multiple values')
        else:
            mean = sum(data)/len(data)
            sum_ = 0
            for i in data:
                sum_ += (i - mean)**2
            variance = sum_/len(data)
            q = variance/mean
            self.p = 1 - q
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """ Method that calculates the value of the PMF for
            a given number of “successes”

        Args:
            k (int): The number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        fact_n = self.factorial(self.n)
        fact_k = self.factorial(k)
        fact_nk = self.factorial((self.n - k))
        part_1 = fact_n / (fact_k * fact_nk)
        part_2 = (self.p**k) * ((1 - self.p)**(self.n - k))
        return part_1*part_2

    def cdf(self, k):
        """ Method that alculates the value of the CDF for
            a given number of “successes”

        Args:
            k (int): The number of “successes".
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    def factorial(self, value):
        """ Method that calculates factorial of a number

        Args:
            value (int): Value to calculate factorial
        """
        factorial = 1
        for i in range(1, value + 1):
            factorial *= i
        return factorial
