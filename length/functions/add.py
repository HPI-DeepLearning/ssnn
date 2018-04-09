from length.function import Function


class Add(Function):

    name = "Add"

    def internal_forward(self, inputs):
        x, y = inputs
        return x + y

    def internal_backward(self, inputs, gradients):
        gradient, = gradients
        return gradient, gradient


def add(x, y):
    return Add()(x, y)
