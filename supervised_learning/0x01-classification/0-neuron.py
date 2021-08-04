#!/usr/bin/env python3
"""  Neuron Class
"""
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing
        binary classification.
    """
    def __init__(self, nx):
        """ Constructor
        Args:
            nx (int): Number of input features to the neuron.
        Attributes:
            W: The weights vector for the neuron.
            b: The bias for the neuron.
            A: The activated output of the neuron (prediction).
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 10, (1, nx))
        self.b = 0
        self.A = 0
