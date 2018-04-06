from length.optimizer import Optimizer


class SGD(Optimizer):
    """
    An optimizer that does plain Stochastic Gradient Descent
    """

    def __init__(self, lr):
        self.lr = lr

    def run_update_rule(self, gradients, _):
        param_deltas = [self.lr * gradient for gradient in gradients]
        return param_deltas
