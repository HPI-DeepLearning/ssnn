from data_sets.mnist_like import MnistLike


class FashionMnist(MnistLike):
    def __init__(self, batch_size, shuffle=True, repeat=True):
        super().__init__("fashion-mnist.s3-website.eu-central-1.amazonaws.com/", "fashion", batch_size, shuffle, repeat)
