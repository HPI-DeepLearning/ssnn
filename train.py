import argparse

import length.functions as F

from length.data_sets import Mnist
from length.graph import Graph
from length.layers import FullyConnected
from length.optimizers import Adam


def main(args):
    data_set = Mnist(64)

    fully_connected_1 = FullyConnected(784, 100)
    fully_connected_2 = FullyConnected(100, 100)
    fully_connected_3 = FullyConnected(100, 10)
    optimizer = Adam(0.001)

    for epoch in range(args.num_epochs):
        for iteration, batch in enumerate(data_set.train):
            data_graph = Graph(batch.data)
            label_graph = Graph(batch.labels)

            h = F.relu(fully_connected_1(data_graph))
            h = F.relu(fully_connected_2(h))
            h = fully_connected_3(h)
            loss = F.softmax_cross_entropy(h, label_graph)

            loss.backward(optimizer)

            if iteration % 50 == 0:
                accuracy = F.accuracy(h, label_graph)
                print(epoch, iteration, loss.data, accuracy.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")

    main(parser.parse_args())
