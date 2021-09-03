#!/usr/bin/env python3
""" Functions:
        pool(images, kernel_shape, stride, mode='max')
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ Function that performs a same convolution on grayscale
        images.

    Args:
        images (Numpy array): Numpy.ndarray with shape (m, h, w, c)
            containing multiple images.
            m is the number of images.
            h is the height in pixels of the images.
            w is the width in pixels of the images.
            c is the number of channels in the image.
        kernel_shape (Numpy array): Numpy.ndarray with shape (kh, kw)
            containing the kernel shape for the pooling
            kh is the height of the kernel.
            kw is the width of the kernel.
        stride (tuple): Tuple of (sh, sw).
            sh is the stride for the height of the image.
            sw is the stride for the width of the image.
        mode (str): Indicates the type of pooling.
            max indicates max pooling.
            avg indicates average pooling.
    Returns:
        A numpy.ndarray containing the pooled images.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape.shape[0]
    kw = kernel_shape.shape[1]
    sh = stride[0]
    sw = stride[1]

    oh = int(((h - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)
    output = np.zeros((m, oh, ow, c))

    for x in range(oh):
        for y in range(ow):
            x0 = x * sh
            y0 = y * sw
            x1 = x0 + kh
            y1 = y0 + kw
            max_pool = np.max
            if mode == 'avg':
                max_pool = np.mean
            output[:, x, y] = max_pool(images[:, x0:x1, y0:y1],
                                       axis=(1, 2))
    return output
