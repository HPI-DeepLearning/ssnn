import argparse

import length.functions as F

from length.data_sets import Mnist
from length.models import Mlp
from length.optimizers import Adam


def main(args):
    data_set = Mnist(64)

    model = Mlp()
    optimizer = Adam(0.001)

    for epoch in range(args.num_epochs):
        for iteration, batch in enumerate(data_set.train):
            model.forward(batch)
            model.backward(optimizer)

            if iteration % 50 == 0:
                accuracy = F.accuracy(model.predictions, batch.labels).data
                print("train: epoch: {:02d}, loss: {:05.2f}, accuracy {:.2f}, iteration: {:03d}".
                      format(epoch, model.loss.data, accuracy, iteration))

        print("running test set...")
        sum_accuracy = 0.0
        sum_loss = 0.0
        for iterations, batch in enumerate(data_set.test):
            model.forward(batch, train=False)
            sum_accuracy += F.accuracy(model.predictions, batch.labels).data
            sum_loss += model.loss.data
        nr_batches = iterations - 1
        print(" test: epoch: {:02d}, loss: {:05.2f}, accuracy {:.2f}".
              format(epoch, sum_loss / nr_batches, sum_accuracy / nr_batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training")

    main(parser.parse_args())
