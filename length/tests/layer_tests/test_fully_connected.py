import numpy as np

from length import constants
from length.graph import Graph
from length.layers import FullyConnected
from length.tests import gradient_checker


def test_fully_connected_forward():
    data = np.random.uniform(-1, 1, (10, 50)).astype(constants.DTYPE)

    layer = FullyConnected(50, 20)
    layer_output = layer(Graph(data))
    gradient_checker.assert_allclose(layer_output.data, np.dot(data, layer.weights.T) + layer.bias)


def test_fully_connected_backward():
    data = np.random.uniform(-1, 1, (10, 50)).astype(constants.DTYPE)

    gradient = np.full((10, 20), 2, dtype=constants.DTYPE)

    layer = FullyConnected(50, 20)
    comp_grad_x, comp_grad_weight, comp_grad_bias = layer.internal_backward((data,), (gradient,))

    f = lambda: layer.internal_forward((data,))
    num_grad_x, num_grad_weight, num_grad_bias = gradient_checker.compute_numerical_gradient(f, (data, layer.weights, layer.bias), (gradient,), eps=1e-2)

    gradient_checker.assert_allclose(comp_grad_x, num_grad_x, atol=1e-4)
    gradient_checker.assert_allclose(comp_grad_weight, num_grad_weight, atol=1e-4)
    gradient_checker.assert_allclose(comp_grad_bias, num_grad_bias, atol=1e-4)


def test_initialization():
    layer = FullyConnected(50, 42)
    assert not (layer.weights == 0).all()
