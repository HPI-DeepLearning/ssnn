import numpy as np

from length.abstract_initializer import AbstractInitializer


class Xavier(AbstractInitializer):
    """
    Xavier initializer
    """

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, array):
        s = self.scale * np.sqrt(2. / (array.shape[0] + array.shape[1]))
        array[...] = np.random.normal(loc=0.0, scale=s, size=array.shape)
