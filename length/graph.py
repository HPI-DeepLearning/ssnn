import numpy as np


class Graph:
    """
    Graph stores data and computational history of the neural network
    """
    def __init__(self, data, predecessors=None, creator=None):
        self.predecessors = predecessors
        self.data = data
        self.creator = creator
        self.grad = None

    def backward(self, optimizer):
        if self.creator is None:
            # early exit if we are at the top of the computational graph
            return

        # if the size of the data array is 1, we are at the bottom of the computational graph
        # so, we are starting with a gradient of 1
        if self.data.size == 1:
            self.grad = np.ones_like(self.data)

        candidate_layers = []
        seen_layers = set()

        def add_candidate_layer(candidate):
            if candidate is not None and candidate not in seen_layers:
                candidate_layers.append(candidate)
                seen_layers.add(candidate)

        add_candidate_layer(self)

        while candidate_layers:
            candidate_layer = candidate_layers.pop()
            if candidate_layer.creator is None:
                continue

            if candidate_layer.creator.needs_optimizer:
                candidate_layer.creator.optimizer = optimizer

            gradients = candidate_layer.creator.backward(candidate_layer.grad)

            for predecessor, gradient in zip(candidate_layer.predecessors, gradients):
                predecessor.grad = gradient
                if gradient is not None:
                    # the gradient flows to another layer (does not happen with loss layers)
                    add_candidate_layer(predecessor)





