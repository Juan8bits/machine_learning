#!/usr/bin/env python3
"""  Neural Network Class
"""
import numpy as np


class NeuralNetwork:
    """ Defines a neural network with one hidden layer performing
        binary classification.
    """
    def __init__(self, nx, nodes):
        """ Method to instantiates a Neural Network

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method that returns the value of the attribute "W1".
        """
        return self.__W1

    @property
    def b1(self):
        """ Getter method that return the value of the attribute "b1".
        """
        return self.__b1

    @property
    def A1(self):
        """ Getter method that return the value of the attribute "A1".
        """
        return self.__A1

    @property
    def W2(self):
        """ Getter method that return the value of the attribute "W2".
        """
        return self.__W2

    @property
    def b2(self):
        """ Getter method that return the value of the attribute "b2".
        """
        return self.__b2

    @property
    def A2(self):
        """ Getter method that return the value of the attribute "A2".
        """
        return self.__A2
