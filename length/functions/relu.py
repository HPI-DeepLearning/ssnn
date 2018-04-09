import numpy as np

from length.function import Function


class Relu(Function):
    """
    The Relu Layer is a non-linear activation
    """

    name = "ReLU"

    def __init__(self):
        super().__init__()
        self.output = None

    def internal_forward(self, inputs):
        x, = inputs
        self.output = np.maximum(x, np.zeros_like(x))
        return self.output,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        copy = self.output.copy()
        copy[copy > 0] = 1
        grad_x = np.multiply(grad_in, copy)
        assert grad_x.shape == x.shape
        return grad_x,


def relu(x):
    return Relu()(x)
