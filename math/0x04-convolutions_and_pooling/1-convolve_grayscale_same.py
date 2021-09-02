#!/usr/bin/env python3
""" Functions:
        convolve_grayscale_same(images, kernel)
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Function that performs a same convolution on grayscale
        images.

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

    # Calculate the right padding for the output
    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    # Creating the output matrix with shape (m, h, w) as the inital input
    output = np.zeros((m, h, w))

    # Pad of zeros around the output matrix.
    pad_img = np.pad(images,
                     pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)

    for x in range(w):
        for y in range(h):
            x1 = x + kw
            y1 = y + kh
            output[:, y, x] = np.sum(pad_img[:, y:y1, x:x1] * kernel,
                                     axis=(1, 2))

    return output
