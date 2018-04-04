import numpy as np

from length.abstract_layer import AbstractLayer


class Sum(AbstractLayer):
    """
    Abstract Layer is a super class for all neural network layers
    """

    def __init__(self):
        super().__init__()

    def internal_forward(self, inputs):
        x, = inputs
        return np.sum(x),

    def internal_backward(self, inputs, gradients):
        x, = inputs
        grad_in, = gradients
        grad_x = np.full_like(x, grad_in)
        assert grad_x.shape == x.shape
        return grad_x,

    def internal_update(self, parameter_deltas):
        pass

    def __call__(self, x):
        return self.forward((x,))
