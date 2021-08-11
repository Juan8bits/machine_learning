#!/usr/bin/env python3
"""  Neural Network Class
"""
import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing
        binary classification.
    """
    def __init__(self, nx, layers):
        """ Constructor method that instanciate a new
            DeepNeuralNetwork object.

        Args:
            nx (int): The number of input features.
            layers (list): List representing the number of
                nodes in each layer of the network.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not layers:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(self.L):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if lay == 0:
                He = (np.random.randn(layers[lay], nx)
                      * np.sqrt(2 / nx))
                self.weights["W{}".format(lay + 1)] = He
            else:
                He = (np.random.randn(layers[lay], layers[lay - 1])
                      * np.sqrt(2 / layers[lay - 1]))
                self.weights["W{}".format(lay + 1)] = He
            # Zero initialization for biases
            self.weights["b{}".format(lay + 1)] = np.zeros((layers[lay], 1))

    @property
    def L(self):
        """ Getter method to private variable L.
        """
        return self.__L

    @property
    def cache(self):
        """ Getter method to private variable cache.
        """
        return self.__cache

    @property
    def weights(self):
        """ Getter method to private variable weights.
        """
        return self.__weights
