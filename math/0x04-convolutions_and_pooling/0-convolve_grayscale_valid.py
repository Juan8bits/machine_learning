#!/usr/bin/env python3
""" Functions:
        convolve_grayscale_valid(images, kernel)
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Function that performs a valid convolution on grayscale
        images

    Args:
        images (Numpy array): Numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
        kernel (Numpy array): Numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
            kh is the height of the kernel.
            kw is the width of the kernel.
    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Output matrix height and width
    oh = h - kh + 1
    ow = w - kw + 1

    # Creating the output matrix with shape (m, oh, ow)
    output = np.zeros((m, oh, ow))

    # Loop over every pixel in the output
    for x in range(ow):
        for y in range(oh):
            x1 = x + kh
            y1 = y + kw
            output[:, y, x] = np.sum(images[:, y:y1, x:x1] * kernel,
                                     axis=(1, 2))

    return output
