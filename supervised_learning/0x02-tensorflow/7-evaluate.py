#!/usr/bin/env python3
""" Functions:
        def evaluate(X, Y, save_path)
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """ Function that evaluates the output of a neural network.
    Args:
        - X(numpy object): is a numpy.ndarray containing the
            input data to evaluate.
        - Y(numpy object): is a numpy.ndarray containing the
            one-hot labels for X.
        - save_path(str): Location to load the model from.
    Return:
        The network’s prediction, accuracy, and loss, respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        eval_y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        eval_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        eval_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return eval_y_pred, eval_accuracy, eval_loss
