from .mnist_like import MNISTLike


class MNIST(MNISTLike):
    """
    The hand-written digit data set
    """

    name = "MNIST"
    url = "http://yann.lecun.com/exdb/mnist/"
