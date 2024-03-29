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
        preactivation_1 = (self.W1 @ X) + self.b1
        activation_1 = 1 / (1 + np.exp(-preactivation_1))
        preactivation_2 = (self.W2 @ activation_1) + self.b2
        activation_2 = 1 / (1 + np.exp(-preactivation_2))
        self.__A1 = activation_1
        self.__A2 = activation_2
        return self.A1, self.A2

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

    def evaluate(self, X, Y):
        """ Method that evaluates the neuron's predictions.

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
        """
        A = self.forward_prop(X)
        J = self.cost(Y, A[1])
        A = np.where(A[1] >= 0.5, 1, 0)
        return A, J

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Method that calculates one pass of gradient descent on the neuron.

            dz2 = dJ/dz = A2 - Y
            dW2 = dJ/dW = 1/m * dz2*A1(transpose)
            db2 = 1/m * dz2

            dz1 = dJ/dz = W2(transpose) * dz2 * A1(1 - A1)
            dw1 = dJ/dW = 1/m * dz1*X(tranpose)
            db1 = 1/m * dz1

            gradient descent -> θ = θ - α ▼F

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
            A1 (numpy object): The output of the hidden layer.
            A2 (numpy object): The predicted output.
            alpha (float, optional):  Is the learning rate. Defaults to 0.05.
        """
        # Back propagation derivates.
        dz2 = A2 - Y
        dW2 = (1/Y.shape[1]) * np.matmul(dz2, A1.T)
        db2 = np.mean(dz2)

        dz1 = np.matmul(self.W2.T, dz2) * A1 * (1 - A1)
        dW1 = (1/Y.shape[1]) * np.matmul(dz1, X.T)
        db1 = np.mean(dz1)

        # Applying gradiant descent for layers
        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = np.full((1, 1), self.b2 - alpha * db2)
        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1
