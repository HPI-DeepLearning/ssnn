from data_sets.mnist_like import MnistLike


class Mnist(MnistLike):
    """
    The hand-written digit data set
    """

    name = "mnist"
    url = "http://yann.lecun.com/exdb/mnist/"
