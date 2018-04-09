import numpy as np

from length.function import Function


class Dropout(Function):

    name = "Dropout"

    def __init__(self, dropout_ratio):
        super().__init__()
        if not 0.0 <= dropout_ratio < 1:
            raise ValueError("dropout_ratio must be in range [0, 1)")
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def internal_forward(self, inputs):
        x, = inputs
        scale = x.dtype.type(1. / (1 - self.dropout_ratio))
        flag = np.random.rand(*x.shape) >= self.dropout_ratio
        self.mask = scale * flag
        return x * self.mask,

    def internal_backward(self, inputs, gradients):
        gradient, = gradients
        return gradient * self.mask,


def dropout(x, dropout_ratio=0.5, train=True):
    if train:
        return Dropout(dropout_ratio)(x)
    return x
