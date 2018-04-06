import numpy as np

from length.layer import Layer
from length.optimizers import Adam
from length.tests import gradient_checker


def test_adam():
    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    gradients = np.random.random((10, 400))

    layer = Layer()

    deltas, = optimizer.run_update_rule((gradients,), layer)
    np.testing.assert_array_less(deltas, np.full_like(gradients, learning_rate))


def test_adam_increasing_lr():
    learning_rate = 0.001
    optimizer = Adam(learning_rate)

    gradients = np.random.random((10, 400)) / 100

    layer = Layer()

    for i in range(0, 5):
        _, = optimizer.run_update_rule((gradients,), layer)

    inverse_gradients = -1 * gradients

    current_deltas, = optimizer.run_update_rule((inverse_gradients,), layer)

    np.testing.assert_array_less(np.zeros_like(current_deltas), current_deltas)

    for i in range(0, 5):
        current_deltas, = optimizer.run_update_rule((inverse_gradients,), layer)

    np.testing.assert_array_less(current_deltas, np.zeros_like(current_deltas))
