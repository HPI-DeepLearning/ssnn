from data_sets.mnist_like import MnistLike


class Mnist(MnistLike):
    def __init__(self, batch_size, **kwargs):
        super().__init__("http://yann.lecun.com/exdb/mnist/", "mnist", batch_size, **kwargs)
