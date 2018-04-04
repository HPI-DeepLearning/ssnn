

class AbstractLayer:
    """
    Abstract Layer is a super class for all neural network layers
    """
    def __init__(self):
        pass

    def forward(self, inputs):
        """
        inputs is a tuple of numpy arrays
        :returns outputs as tuple of numpy arrays
        """
        raise NotImplementedError

    def backward(self, inputs, gradients):
        """
        :param inputs: is a tuple of numpy arrays
        :param gradients: is a tuple of numpy arrays
        :returns gradients as tuple of numpy arrays
        """
        raise NotImplementedError

