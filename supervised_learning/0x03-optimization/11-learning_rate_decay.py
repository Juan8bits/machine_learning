#!/usr/bin/env python3
""" Functions:
        learning_rate_decay(alpha, decay_rate, global_step, decay_step)
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Functions that updates the learning rate using inverse
        time decay in numpy.

    Args:
        alpha (): is the original learning rate.
        decay_rate (): is the weight used to find the rate at which
            a will decay.
        global_step (): is the # of passes of gradient descent that
            have elapsed.
        decay_step (): is the number of passes of gradient descent that should.
            occur before alpha is decayed further.

    Return:
        The updated value for alpha.
    """
    dr = decay_rate
    updated_dr = alpha / (1 + dr * int(global_step / decay_step))
    return updated_dr
