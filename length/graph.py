

class Graph:
    """
    Graph stores data and computational history of the neural network
    """
    def __init__(self, data, predecessors=None, creator=None):
        self.predecessors = predecessors
        self.data = data
        self.creator = creator

    def backward(self, optimizer):
        pass
