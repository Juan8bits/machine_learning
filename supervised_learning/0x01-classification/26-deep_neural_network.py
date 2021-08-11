#!/usr/bin/env python3
"""  Neural Network Class
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle


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

    def evaluate(self, X, Y):
        """ Method that evaluates the neuron's predictions.

        Args:
            X (numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data.
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
        """
        # A = (FwdPropagation last layer, all FwdPropagations)
        A = self.forward_prop(X)
        J = self.cost(Y, A[0])
        A = np.where(A[0] >= 0.5, 1, 0)
        return A, J

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Method that calculates one pass of gradient descent on the neuron.

            dz2 = dJ/dz = A2 - Y
            dW2 = dJ/dW = 1/m * dz2*A1(transpose)
            db2 = 1/m * dz2

            dz1 = dJ/dz = W2(transpose) * dz2 * A1(1 - A1)
            dw1 = dJ/dW = 1/m * dz1*X(tranpose)
            db1 = 1/m * dz1

            gradient descent -> θ = θ - α ▼F

        Args:
            Y (numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
            cache (dic): Dictionary containing all the intermediary
                values of the network.
            alpha (float, optional):  Is the learning rate. Defaults to 0.05.
        """
        # Derivation for the last layer (dJ/dz)
        dz = self.cache["A{}".format(self.L)] - Y

        for layer in range(self.L, 0, -1):
            weights = self.__weights["W{}".format(layer)]
            A = self.__cache["A{}".format(layer - 1)]
            bias = self.__weights["b{}".format(layer)]

            # Back propagation derivates.
            dW = (1/Y.shape[1]) * np.matmul(dz, A.T)
            db = np.sum(dz, axis=1, keepdims=True) / Y.shape[1]
            dz = np.matmul(weights.T, dz) * A * (1 - A)

            # Applying gradiant descent for layers
            self.__weights["W{}".format(layer)] = weights - alpha * dW
            self.__weights["b{}".format(layer)] = bias - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Method to trains the neuron.
            The model should change the weights and bias given,
            according to the backpropagation(gradiant descent),
            so first, all information pass by forward propagation
            algorithm wich is our classification algorithm (sigmoidea).
        Args:
            X (Numpy object): Numpy.ndarray with shape (nx, m) that
                contains the input data
            Y (Numpy object): Numpy.ndarray with shape (1, m) that
                contains the correct labels for the input data.
            verbose (Bool): Variable that defines whether or not to
                print information about the training.
            graph (Bool): Variable that that defines whether or not
                to graph information about the training once the
                training has completed.
            step (int): step iterations for a graph.
            iterations (int, optional): Is the number of iterations
                to train over. Defaults to 5000.
            alpha (float, optional): Learning rate. Defaults to 0.05.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if not isinstance(step, int):
            raise TypeError('step must be an integer')
        if step < 0 or step > iterations:
            raise ValueError('step must be positive and <= iterations')

        cost_per_iter = []
        iters = []
        for i in range(iterations + 1):
            A, J = self.evaluate(X, Y)
            self.gradient_descent(Y, self.cache, alpha)
            if verbose is True and i % step == 0:
                cost_per_iter.append(J)
                iters.append(i)
                print("Cost after {} iterations: {}".format(i, J))
        if graph is True and verbose is True:
            plt.plot(iters, cost_per_iter, '-b')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return A, J

    def save(self, filename):
        """ Method that saves the instance object to a file
            in pickle format.

        Args:
            filename (Filedescriptor): Is the file to which
            the object should be saved.
        """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)
            fd.close()

    @staticmethod
    def load(filename):
        """ Method that loads a pickled DeepNeuralNetwork object.

        Args:
            filename (Filedescriptor): Is the file from which
            the object should be loaded.
        Return:
            The loaded object, or None if filename doesn’t exist.
        """
        try:
            with open(filename, 'rb') as fd:
                obj = pickle.load(fd)
                return obj
        except FileNotFoundError:
            return None
