import numpy as np
import os
from PIL import Image

from length.constants import DTYPE
from length.data_sets import Mnist


def test_mnist_without_scale():
    image = Image.open(os.path.join("length", "tests", "data", "5.png"))
    expected = np.array(image)
    data_set = Mnist(1, scale=None)
    for batch in data_set.train:
        first_sample, first_label = batch.data.data[0], batch.labels.data[0]
        np.testing.assert_equal(expected, first_sample.reshape(28, 28))
        assert first_label == 5
        break


def test_mnist_reading_test():
    data_set = Mnist(100)
    iterations = 0
    for batch in data_set.train:
        assert (100, 784) == batch.data.shape
        assert DTYPE == batch.data.data.dtype
        assert (100,) == batch.labels.shape
        assert np.uint8 == batch.labels.data.dtype
        iterations += 1
    assert iterations == 600


def test_mnist_reading_train():
    data_set = Mnist(100)
    iterations = 0
    for batch in data_set.test:
        assert (100, 784) == batch.data.shape
        assert DTYPE == batch.data.data.dtype
        assert (100,) == batch.labels.shape
        assert np.uint8 == batch.labels.data.dtype
        iterations += 1
    assert iterations == 100


def test_mnist_reading_2_dimensions():
    data_set = Mnist(100, 2)
    iterations = 0
    for batch in data_set.train:
        assert (100, 28, 28) == batch.data.shape
        assert DTYPE == batch.data.data.dtype
        assert (100,) == batch.labels.shape
        assert np.uint8 == batch.labels.data.dtype
        iterations += 1
    assert iterations == 600


def test_mnist_reading_3_dimensions():
    data_set = Mnist(100, 3)
    iterations = 0
    for batch in data_set.train:
        assert (100, 1, 28, 28) == batch.data.shape
        assert DTYPE == batch.data.data.dtype
        assert (100,) == batch.labels.shape
        assert np.uint8 == batch.labels.data.dtype
        iterations += 1
    assert iterations == 600
