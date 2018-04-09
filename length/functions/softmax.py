import numpy as np

from length.function import Function


def log_softmax(x):
    def logsumexp(x):
        m = x.max(axis=1, keepdims=True)
        y = x - m
        np.exp(y, out=y)
        s = y.sum(axis=1, keepdims=True)
        np.log(s, out=s)
        m += s
        return m

    log_z = logsumexp(x)
    # log(e^(x)/log_z)
    y = x - log_z
    return y


class Softmax(Function):
    """
    Abstract Layer is a super class for all neural network layers
    """

    name = "Softmax"

    def __init__(self):
        super().__init__()
        self.y = None

    def internal_forward(self, inputs):
        x, = inputs
        assert x.ndim == 2, "Softmax only supports two-dimensional input"
        self.y = np.exp(log_softmax(x))
        return self.y,

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = self.y * grad_in
        sum_grad_x = grad_x.sum(axis=1, keepdims=True)
        grad_x -= self.y * sum_grad_x

        assert x.shape == grad_x.shape

        return grad_x,


def softmax(x):
    return Softmax()(x)
