import argparse

from length.graph import Graph
from length.layers.fully_connected import FullyConnectedLayer


def main(args):
    data_loader = DataLoader("mnist")

    fully_connected_1 = FullyConnectedLayer(784, 256)

    fully_connected_2 = FullyConnectedLayer(256, 512)

    loss_layer = SoftMaxCrossEntropy()

    optimizer = SGD()

    for epoch in range(args.num_epochs):
        for iteration, batch in enumerate(data_loader):
            data, label = batch

            data_graph = Graph(None, data)
            label_graph = Graph(None, label)

            h = fully_connected_1.forward(data_graph)
            h = fully_connected_2.forward(h)
            loss = loss_layer.forward(h, label_graph)

            loss.backward(optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")

    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")

    main(parser.parse_args())
