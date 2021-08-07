#!/usr/bin/env python3
"""  Neuron Class
"""
import numpy as np


class Neuron:
    """ Class that defines a single neuron performing
        binary classification.
    """
    e = 2.7182818285

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
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter method that return the value of the attribute "W".
        """
        return self.__W

    @property
    def A(self):
        """ Getter method that return the value of the attribute "A".
        """
        return self.__A

    @property
    def b(self):
        """ Getter method that return the value of the attribute "A".
        """
        return self.__b

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
        """
        # preactivation = np.matmul(self.__W, X) + self.__b
        preactivation = (self.W @ X) + self.b
        activation = 1 / (1 + np.exp(-preactivation))
        self.__A = activation
        return self.A

    def cost(self, Y, A):
        """ Method that calculates the cost of the model using
            logistic regression.

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

    def evaluate(self, X, Y):
        """ Method that evaluates the neuron's predictions.

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
        """
        A = self.forward_prop(X)
        J = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, J

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Method that calculates one pass of gradient descent on the neuron.

            dz = dJ/dz = A - Y
            dW = dJ/dW = X*dz
            db = 1/m * dz

            gradient descent = θ = θ - α ▼F

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
            A (numpy object): Numpy.ndarray with shape (1, m) containing
                the activated output of the neuron for each example
            alpha (float, optional):  Is the learning rate. Defaults to 0.05.
        """
        # Back propagation derivates.
        dz = A - Y
        dW = (1/Y.shape[1]) * np.matmul(dz, X.T)
        db = np.mean(dz)
        # Aplying gradiant descent
        self.__W = self.W - alpha * dW
        self.__b = self.b - alpha * db
