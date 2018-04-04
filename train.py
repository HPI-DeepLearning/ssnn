import argparse
import numpy as np

from length.graph import Graph
from length.layers.fully_connected import FullyConnectedLayer
from length.layers.sum import Sum
from length.optimizers.sgd import SGD


def main(args):
    # data_loader = DataLoader("mnist")
    data = np.random.random((10, 784))

    fully_connected_1 = FullyConnectedLayer(784, 100)
    fully_connected_2 = FullyConnectedLayer(100, 100)
    fully_connected_3 = FullyConnectedLayer(100, 10)
    loss_layer = Sum()
    optimizer = SGD(0.001)

    for epoch in range(args.num_epochs):
        for iteration in range(10):

            data_graph = Graph(data)

            h = fully_connected_1(data_graph)
            h = fully_connected_2(h)
            h = fully_connected_3(h)
            loss = loss_layer(h)

            loss.backward(optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")

    main(parser.parse_args())
