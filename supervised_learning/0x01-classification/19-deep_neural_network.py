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

    def forward_prop(self, X):
        """ Method that Calculates the forward propagation of the neuron.

        preactivation: y = mx + b
        activation with sigmoidea:
        1 / (1 + e ^(-x))

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
            contains the input data.
                - nx is the number of input features to the neuron.
                - m is the number of examples.
        Return:
            The output of the neural network and the cache, respectively.
        """

        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights["W{}".format(layer + 1)]
            a_ = self.__cache["A{}".format(layer)]
            biases = self.__weights["b{}".format(layer + 1)]

            # Preactivation. Propagation function.
            preactivation = (weights @ a_) + biases
            # Activation sigmoidea function.
            activation = 1 / (1 + np.exp(-preactivation))
            # Save forward propagation values.
            self.__cache["A{}".format(layer + 1)] = activation

        return self.__cache["A{}".format(layer + 1)], self.__cache

    def cost(self, Y, A):
        """ Method that calculates the cost of the Neural network
            using logistic regression.

        J=−1/m * ∑mi=1 y(i)log(a(i))+(1−y(i))log(1−a(i))
        Args:
            Y (numpy object): Numpy.ndarray with shape (1, m) that contains the
                correct labels for the input data
            A (numpy object): Numpy.ndarray with shape (1, m) containing the
                activated output of the neuron for each example
        """
        m = Y.shape[1]
        j = -(1/m)*np.sum(Y*np.log(A) + (1-Y) * np.log(1.0000001-A))
        return j
