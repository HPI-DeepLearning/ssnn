from length.abstract_optimizer import AbstractOptimizer


class SGD(AbstractOptimizer):

    def __init__(self, lr):
        self.lr = lr

    def run_update_rule(self, gradients):
        param_deltas = [self.lr * gradient for gradient in gradients]
        return param_deltas
