import numpy as np

from length.function import Function
from length.functions.softmax import Softmax


class SoftmaxCrossEntropy(Function):

    def __init__(self):
        super().__init__()
        self.softmax_layer = Softmax()
        self.y = None

    def internal_forward(self, inputs):
        x, t = inputs
        self.y, = self.softmax_layer.internal_forward((x,))
        loss = -np.log(self.y[range(len(t)), t]).sum(keepdims=True) / t.size
        return loss,

    def internal_backward(self, inputs, gradients):
        x, t = inputs
        grad_in, = gradients
        grad_x = self.y.copy()
        grad_x[range(len(t)), t] -= 1
        grad_x *= grad_in[0] / t.size
        return grad_x, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
