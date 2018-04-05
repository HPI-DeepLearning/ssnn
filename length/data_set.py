from collections import namedtuple


Batch = namedtuple("Batch", ("data", "labels"))


class DataSet:
    """
    Abstract DataSet, it should implement ways to retrieve test and train data and support shuffling
    """
    def __init__(self, batch_size, shuffle=True, repeat=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat

    @property
    def test(self):
        raise NotImplementedError()

    @property
    def train(self):
        raise NotImplementedError()
