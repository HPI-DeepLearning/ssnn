from length.data_sets import Mnist


def test_mnist_reading_test():
    data_set = Mnist(100)
    iterations = 0
    for batch in data_set.train:
        assert (100, 784) == batch.data.shape
        assert (100,) == batch.labels.shape
        iterations += 1
    assert iterations == 600


def test_mnist_reading_train():
    data_set = Mnist(100)
    iterations = 0
    for batch in data_set.test:
        assert (100, 784) == batch.data.shape
        assert (100,) == batch.labels.shape
        iterations += 1
    assert iterations == 100


def test_mnist_reading_2_dimensions():
    data_set = Mnist(100, 2)
    iterations = 0
    for batch in data_set.train:
        assert (100, 28, 28) == batch.data.shape
        assert (100,) == batch.labels.shape
        iterations += 1
    assert iterations == 600


def test_mnist_reading_3_dimensions():
    data_set = Mnist(100, 3)
    iterations = 0
    for batch in data_set.train:
        assert (100, 1, 28, 28) == batch.data.shape
        assert (100,) == batch.labels.shape
        iterations += 1
    assert iterations == 600
