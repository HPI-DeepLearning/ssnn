import numpy as np

from length.function import Function
from length.functions.softmax import Softmax


class Accuracy(Function):

    name = "Accuracy"

    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def internal_forward(self, inputs):
        x, t = inputs
        softmaxed_x, = self.softmax.internal_forward((x,))
        predicted_classes = np.argmax(softmaxed_x, axis=1)

        num_correct = (predicted_classes == t).sum()

        return num_correct / len(t),

    def internal_backward(self, inputs, gradients):
        return None, None


def accuracy(x, t):
    return Accuracy()(x, t)
