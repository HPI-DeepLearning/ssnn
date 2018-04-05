from .mnist_like import MnistLike


class FashionMnist(MnistLike):
    """
    The replacement of the hand-written digit data set with fashion items
    """

    name = "fashion"
    url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
