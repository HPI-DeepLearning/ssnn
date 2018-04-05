import numpy as np

from length.function import Function


class Softmax(Function):
    """
    Abstract Layer is a super class for all neural network layers
    """

    def __init__(self):
        super().__init__()
        self.y = None

    def internal_forward(self, inputs):
        x, = inputs
        assert x.ndim == 2, "Softmax only supports two-dimensional input"
        self.y = x - np.amax(x, axis=1, keepdims=True)
        np.exp(self.y, out=self.y)
        self.y /= self.y.sum(axis=1, keepdims=True)
        return self.y,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = self.y * grad_in
        sum_grad_x = grad_x.sum(axis=1, keepdims=True)
        grad_x -= self.y * sum_grad_x

        assert x.shape == grad_x.shape

        return grad_x,
