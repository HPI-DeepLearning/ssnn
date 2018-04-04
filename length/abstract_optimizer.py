

class AbstractOptimizer:

    def run_update_rule(self, gradients):
        raise NotImplementedError
