import numpy as np

from length import constants
from length.functions import Sum
from length.graph import Graph
from length.tests.gradient_checker import compute_numerical_gradient, assert_allclose


def test_sum_forward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)

    sum_function = Sum()
    sum_output = sum_function(Graph(data))
    np.testing.assert_allclose(sum_output.data, data.sum())


def test_sum_backward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)
    gradient = np.array([2], dtype=constants.DTYPE)

    data_graph = Graph(data)
    sum_function = Sum()
    sum_function(data_graph)
    computed_gradients, = sum_function.backward((gradient,))

    f = lambda: sum_function.internal_forward((data,))
    numerical_gradients, = compute_numerical_gradient(f, (data,), (gradient,))

    assert_allclose(computed_gradients, numerical_gradients, atol=1e-4, rtol=1e-3)
