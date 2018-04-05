import shutil
import pytest

from data_sets.mnist import Mnist


temp_folder = ".temp"


@pytest.fixture(scope="module")
def cleanup():
    yield cleanup
    print("delete temp folder")
    shutil.rmtree(temp_folder)


def test_download(cleanup):
    data_set = Mnist(10)
    data_set.path = temp_folder
    for batch in data_set:
        print("!")


