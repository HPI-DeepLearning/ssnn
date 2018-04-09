import numpy as np

from length import constants
from length.functions import sigmoid
from length.functions.sigmoid import Sigmoid
from length.graph import Graph
from length.tests import gradient_checker


def test_sigmoid_forward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)

    sigmoid_output = sigmoid(Graph(data))
    desired = np.divide(np.ones_like(data), np.add(np.ones_like(data), np.exp(np.negative(data))))
    np.testing.assert_allclose(sigmoid_output.data, desired)


def test_sigmoid_backward():
    data = np.random.uniform(-1, 1, (3, 2)).astype(constants.DTYPE)
    gradient = np.random.random(data.shape).astype(constants.DTYPE)

    data_graph = Graph(data)
    sigmoid_function = Sigmoid()
    sigmoid_function(data_graph)
    computed_gradients, = sigmoid_function.backward(gradient)

    f = lambda: sigmoid_function.internal_forward((data,))
    numerical_gradients, = gradient_checker.compute_numerical_gradient(f, (data,), (gradient,))

    gradient_checker.assert_allclose(computed_gradients, numerical_gradients, atol=1e-4, rtol=1e-3)
