import argparse
import numpy as np

from length.graph import Graph
from length.layers.fully_connected import FullyConnected
from length.functions.softmax_cross_entropy import SoftmaxCrossEntropy
from length.optimizers.sgd import SGD


def main(args):
    # data_loader = DataLoader("mnist")
    data = np.random.random((10, 784))
    labels = np.zeros((10,), dtype=np.int32)

    fully_connected_1 = FullyConnected(784, 100)
    fully_connected_2 = FullyConnected(100, 100)
    fully_connected_3 = FullyConnected(100, 10)
    loss_layer = SoftmaxCrossEntropy()
    optimizer = SGD(0.001)

    for epoch in range(args.num_epochs):
        for iteration in range(10):

            data_graph = Graph(data)
            label_graph = Graph(labels)

            h = fully_connected_1(data_graph)
            h = fully_connected_2(h)
            h = fully_connected_3(h)
            loss = loss_layer(h, label_graph)

            loss.backward(optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")

    main(parser.parse_args())
