import os
import shutil
import pytest

from data_sets.mnist import Mnist


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
        assert batch.data.shape == (100, 28, 28)
        assert batch.labels.shape == (100,)
        iterations += 1
    assert iterations == 600


def test_mnist_reading_train():
    data_set = Mnist(100)
    iterations = 0
    for batch in data_set.test:
        assert batch.data.shape == (100, 28, 28)
        assert batch.labels.shape == (100,)
        iterations += 1
    assert iterations == 100


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
