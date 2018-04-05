

class Batch:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


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
