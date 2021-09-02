#!/usr/bin/env python3
""" Functions:
        convolve_grayscale_padding(images, kernel)
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding (tuple): Tuple of (ph, pw)
            ph is the padding for the height of the image.
            pw is the padding for the width of the image.
            the image should be padded with 0’s.
    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Padding for the output
    ph = padding[0]
    pw = padding[1]

    # Pad of zeros around the output matrix.
    pad_img = np.pad(images,
                     pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)

    # Output matrix height and width
    oh = int(pad_img.shape[1] - kh + 1)
    ow = int(pad_img.shape[2] - kw + 1)

    # Creating the output matrix with shape (m, h, w) as the inital input
    output = np.zeros((m, oh, ow))


    for x in range(ow):
        for y in range(oh):
            x1 = x + kw
            y1 = y + kh
            output[:, y, x] = np.sum(pad_img[:, y:y1, x:x1] * kernel,
                                     axis=(1, 2))

    return output
