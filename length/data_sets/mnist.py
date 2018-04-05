from data_sets.mnist_like import MnistLike


class Mnist(MnistLike):
    def __init__(self, batch_size, shuffle=True, repeat=True):
        super().__init__("http://yann.lecun.com/exdb/mnist/", "mnist", batch_size, shuffle, repeat)
