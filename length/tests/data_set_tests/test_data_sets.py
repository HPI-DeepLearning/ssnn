import os
import shutil
import pytest

from length.data_sets import Mnist, FashionMnist

temp_folder = ".temp"


@pytest.fixture(scope="module")
def cleanup():
    yield cleanup
    print("delete temp folder")
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)


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


def test_mnist_downloading(cleanup):
    data_set = Mnist(10, delay_loading=True)
    data_set.path = temp_folder
    data_set.download_files()

    required_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for file in required_files:
        assert os.path.isfile(os.path.join(temp_folder, file))


def test_fashion_mnist_downloading(cleanup):
    data_set = FashionMnist(10, delay_loading=True)
    data_set.path = temp_folder
    data_set.download_files()

    required_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for file in required_files:
        assert os.path.isfile(os.path.join(temp_folder, file))
