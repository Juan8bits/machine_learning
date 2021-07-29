#!/usr/bin/env python3
""" Class to represents a normal distribution.
"""


class Normal:
    """ Class to represents a normal distribution.
    """
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Constructor

        Args:
            data (List): List of the data to be used to estimate
                the distribution. Defaults to None.
            mean (): Mean of the distribution. Defaults to 0..
            stddev (float): Standard deviation of the distribution.
                Defaults to 1..
        """
        if data is None:
            if stddev < 1:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            self.mean = float(mean)
        elif not isinstance(data, list):
            raise TypeError('data must be a list')
        elif not len(data) > 1:
            raise ValueError('data must contain multiple values')
        else:
            self.mean = sum(data)/len(data)
            num = 0
            for i in data:
                num += (i - self.mean)**2
            self.stddev = (num/len(data)) ** (1/2)

    def z_score(self, x):
        """ Method that calculates the z-score of a given x-value.

        Args:
            x (int): x value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Method that calculates the x-value of a given z-score.

        Args:
            z (int): z value.
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ Method that calculates the value of the PDF for a
            given x-value
        Args:
            x (int): x value.
        """
        part_1 = 1 / (self.stddev * ((2 * self.pi) ** (1/2)))
        exponent = -(1 / 2) * (((x - self.mean) / self.stddev) ** 2)
        return part_1 * self.e ** exponent

    def cdf(self, x):
        """ Method that calculates the value of the CDF for a
            given x-value

        Args:
            x (int): x value.
        """
        arg = (x - self.mean) / (self.stddev * (2 ** (1/2)))
        erf = (2 / self.pi ** 0.5) * \
              (arg - (arg ** 3) / 3 + (arg ** 5) / 10 -
               (arg ** 7) / 42 + (arg ** 9) / 216)
        return (1/2) * (1 + erf)
