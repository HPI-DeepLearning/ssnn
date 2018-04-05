

class Batch:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


class DataSet:
    def __init__(self, batch_size, shuffle=True, repeat=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
