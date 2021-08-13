#!/usr/bin/env python3
""" Functions:
        calculate_accuracy(y, y_pred)
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ Function that calculates the accuracy of a prediction.

         accuracy = correct_predictions / all_predictions

    Args:
        y (tensor object): Placeholder for the labels of the input data.
        y_pred (tensor object): A tensor containing the network’s predic.
    Return:
        A tensor containing the decimal accuracy of the prediction.
    """
    right_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))
    return accuracy
