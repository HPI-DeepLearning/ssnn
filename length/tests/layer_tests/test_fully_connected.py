import numpy as np

from length.layers.fully_connected import FullyConnected


def test_forward():
    layer = FullyConnected(50, 42)
    random = np.random.random((10, 50))
    result, = layer.internal_forward([random])
    np.testing.assert_array_almost_equal(result, np.dot(random, layer.weights.T) + layer.bias)


def test_initialization():
    layer = FullyConnected(50, 42)
    assert not (layer.weights == 0).all()

