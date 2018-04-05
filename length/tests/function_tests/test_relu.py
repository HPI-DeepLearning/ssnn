import numpy as np

from length import constants
from length.functions import relu
from length.functions.relu import Relu
from length.graph import Graph
from length.tests import gradient_checker


def test_relu_forward():
    data = np.random.uniform(-1, 1, (5, 4)).astype(constants.DTYPE)

    relu_output = relu(Graph(data))
    np.testing.assert_allclose(relu_output.data, np.maximum(data, np.zeros_like(data)))


def test_relu_backward():
    data = np.random.uniform(-1, 1, (5, 4)).astype(constants.DTYPE)
    gradient = np.array([2], dtype=constants.DTYPE)

    data_graph = Graph(data)
    relu_function = Relu()
    relu_function(data_graph)
    computed_gradients, = relu_function.backward((gradient,))

    f = lambda: relu_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,))

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients)
