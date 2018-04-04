from length.graph import Graph


class AbstractLayer:
    """
    Abstract Layer is a super class for all neural network layers
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None

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
        self.inputs = tuple(graph.data for graph in graphs)
        self.outputs = self.internal_forward(self.inputs)

        output_graphs = [Graph(output, predecessors=graphs, creator=self) for output in self.outputs]
        if len(output_graphs) == 1:
            return output_graphs[0]
        return output_graphs

    def backward_and_update(self, gradients, optimizer):
        gradients = self.internal_backward(self.inputs, (gradients,))
        input_gradient = gradients[:len(self.inputs)]
        parameter_gradients = gradients[len(self.inputs):]
        if len(parameter_gradients) > 0:
            parameter_deltas = optimizer.run_update_rule(parameter_gradients)
            self.internal_update(parameter_deltas)
        return input_gradient
