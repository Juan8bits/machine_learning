#!/usr/bin/env python3
""" Functions:
        save_config(network, filename)
        load_config(filename)
"""

import tensorflow.keras as K


def save_config(network, filename):
    """ Function that saves a model’s configuration.
    Args:
        network (Keras object): Is the model whose configuration should
            be saved.
        filename (str): Is the path of the file that the configuration
            should be saved to.
    Returns:
        None.
    """
    json_model = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_model)
    return None


def load_config(filename):
    """ Function that loads a model with a specific configuration.

    Args:
        filename (str): Is the path of the file containing the model’s
          configuration in JSON format.
    Returns:
        The loaded model.
    """
    with open(filename, 'r') as f:
        network_str = f.read()
    return K.models.model_from_json(network_str)
