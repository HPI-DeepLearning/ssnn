from length.graph import Graph


class AbstractLayer:
    """
    Abstract Layer is a super class for all neural network layers
    """
    def __init__(self):
        self.inputs = None

    def internal_forward(self, inputs):
        """
        inputs is a tuple of numpy arrays
        :returns outputs as tuple of numpy arrays
        """
        raise NotImplementedError

    def internal_backward(self, inputs, gradients):
        """
        :param inputs: is a tuple of numpy arrays
        :param gradients: is a tuple of numpy arrays
        :returns gradients as tuple of numpy arrays (first element is gradient with respect to input)
        """
        raise NotImplementedError

    def internal_update(self, parameter_deltas):
        """
        :param parameter_deltas: contains the delta of the parameters to be applied to each parameter
        (same structure as in internal_backward but without the first element)
        """
        raise NotImplementedError

    def forward(self, graphs):
        self.inputs = (graph.data for graph in graphs)
        outputs = self.internal_forward(self.inputs)
        our_graph = Graph(outputs, predecessors=graphs, creator=self)
        return our_graph

    def backward_and_update(self, gradients, optimizer):
        gradients = self.internal_backward(self.inputs, gradients)
        input_gradient = gradients[0]
        parameter_gradients = gradients[1:]
        parameter_deltas = optimizer.run_update_rule(parameter_gradients)
        self.internal_update(parameter_deltas)
        return input_gradient

