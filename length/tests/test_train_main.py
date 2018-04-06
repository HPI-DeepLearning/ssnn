import numpy as np

from length.data_sets import Mnist
from length.models import Mlp
from length.optimizers import Adam, SGD


def test_train():
    np.seterr(divide='raise')

    data_set = Mnist(64)

    for i in range(50):
        model = Mlp()
        optimizer = SGD(0.001)

        for iteration, batch in enumerate(data_set.train):
            model.forward(batch)
            model.backward(optimizer)
            break
