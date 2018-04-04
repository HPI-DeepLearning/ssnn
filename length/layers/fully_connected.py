import numpy as np

from length.abstract_layer import AbstractLayer
from length.constants import DTYPE


class FullyConnectedLayer(AbstractLayer):
    """
    Abstract Layer is a super class for all neural network layers
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.weights = np.zeros((num_inputs, num_outputs,), dtype=DTYPE)
        # Todo: initialize weight matrix
        self.bias = np.zeros((num_outputs,), dtype=DTYPE)

    def forward(self, inputs):
        pass

    def backward(self, inputs, gradients):
        pass

