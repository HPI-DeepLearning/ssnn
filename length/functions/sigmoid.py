import numpy as np

from length.function import Function


class Sigmoid(Function):
    """
    Abstract Layer is a super class for all neural network layers
    """

    name = "Sigmoid"

    def __init__(self):
        super().__init__()
        self.output = None

    def internal_forward(self, inputs):
        x, = inputs
        self.output = 1 / (1 + np.exp(-1 * x))
        return self.output,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = grad_in * self.output * (1 - self.output)
        assert grad_x.shape == x.shape
        return grad_x,


def sigmoid(x):
    return Sigmoid()(x)
