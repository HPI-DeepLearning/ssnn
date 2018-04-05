import numpy as np

from length.optimizers.sgd import SGD
from length.tests.gradient_checker import assert_allclose


def test_sgd():
    learning_rate = 0.001
    optimizer = SGD(learning_rate)

    gradients = np.random.random((10, 400))

    deltas, = optimizer.run_update_rule((gradients,))
    assert_allclose(deltas, gradients * learning_rate)
