import math

import numpy as np

from length import constants
from length.functions import mean_squared_error
from length.functions.mean_squared_error import MeanSquaredError
from length.graph import Graph
from length.tests import gradient_checker


def init(array):
    return np.array(array, dtype=constants.DTYPE)


def fixed_case(with_label=False):
    data = init([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [2, 0, 1, 0],
    ])

    if with_label:
        data_2 = np.array([3, 0, 1], dtype=np.int32)
    else:
        data_2 = init([
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

    return data, data_2


def test_mean_squared_error_forward_zero_loss():
    data = Graph(np.array([0, 0, 0, 1], dtype=constants.DTYPE))
    label = Graph(np.array([0, 0, 0, 1], dtype=constants.DTYPE))

    mse = mean_squared_error(data, label)

    assert float(mse.data) == 0


def test_mean_squared_error_forward_loss():
    data, data_2 = fixed_case()

    mse = mean_squared_error(Graph(data), Graph(data_2))
    assert math.isclose(float(mse.data), 0.583, abs_tol=1e-3)


def test_mean_squared_error_forward_int_input():
    data, labels = fixed_case(with_label=True)

    mse = mean_squared_error(Graph(data), Graph(labels))
    assert math.isclose(float(mse.data), 0.583, abs_tol=1e-3)


def test_mean_squared_error_backward():
    data, data_2 = fixed_case()
    gradients = np.array([2], dtype=constants.DTYPE)

    data_1_graph = Graph(data)
    data_2_graph = Graph(data_2)

    mse_function = MeanSquaredError()
    mse_function(data_1_graph, data_2_graph)
    computed_gradient_1, computed_gradient_2 = mse_function.backward(gradients)

    f = lambda: mse_function.internal_forward((data, data_2))
    numerical_gradient_1, numerical_gradient_2 = gradient_checker.compute_numerical_gradient(
        f,
        (data, data_2),
        (gradients,)
    )

    gradient_checker.assert_allclose(computed_gradient_1, numerical_gradient_1)
    gradient_checker.assert_allclose(computed_gradient_2, numerical_gradient_2)


def test_mean_squared_error_backward_with_label():
    data, data_2 = fixed_case(with_label=True)
    gradients = np.array([2], dtype=constants.DTYPE)

    data_1_graph = Graph(data)
    data_2_graph = Graph(data_2)

    mse_function = MeanSquaredError()
    mse_function(data_1_graph, data_2_graph)
    computed_gradient_1, computed_gradient_2 = mse_function.backward(gradients)
    assert computed_gradient_2 is None

    f = lambda: mse_function.internal_forward((data, data_2))
    numerical_gradient_1, _ = gradient_checker.compute_numerical_gradient(
        f,
        (data, data_2),
        (gradients,)
    )

    gradient_checker.assert_allclose(computed_gradient_1, numerical_gradient_1)
