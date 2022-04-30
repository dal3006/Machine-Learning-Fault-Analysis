#!/usr/bin/env python
"""
Maximum Mean Discrepancy (MMD)

The MMD is implemented as keras regularizer that can be used for
shared layers. This implementation uis tested under keras 1.1.0.

- Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
Advances in neural information processing systems. 2007.

__author__ = "Werner Zellinger"
__copyright__ = "Copyright 2017, Werner Zellinger"
__credits__ = ["Thomas Grubinger, Robert Pollak"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Werner Zellinger"
__email__ = "werner.zellinger@jku.at"
"""

from keras import backend as K
from keras.regularizers import Regularizer
import tensorflow as tf


def mmd(x1, x2, beta):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    x1x1 = gaussian_kernel(x1, x1, beta)
    x1x2 = gaussian_kernel(x1, x2, beta)
    x2x2 = gaussian_kernel(x2, x2, beta)
    diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
    return diff


def gaussian_kernel(x1, x2, beta=1.0):
    r = tf.transpose(x1)
    r = tf.expand_dims(r, 2)
    return tf.reduce_sum(K.exp(-beta * K.square(r - x2)), axis=-1)


# def gaussian_kernel(x1, x2, beta=1.0):
#     r = x1.dimshuffle(0, 'x', 1)
#     return K.exp(-beta * K.square(r - x2).sum(axis=-1))


class MMDRegularizer(Regularizer):
    """
    class structure to use the MMD as activity regularizer of a
    keras shared layer
    """

    def __init__(self, l=1, beta=1.0):
        self.uses_learning_phase = 1
        self.l = l
        self.beta = beta

    def __call__(self, x):
        print(x)
        sim = 0
        # if len(x) < 2:
        #     raise Exception("Need to call in a shared layer")

        # we are in a shared keras layer
        sim = mmd(x[0], x[1], self.beta)
        return K.in_train_phase(self.l * sim, 0)

    def get_config(self):
        # needed for keras layer
        return {'name': self.__class__.__name__,
                'l': float(self.l)}
