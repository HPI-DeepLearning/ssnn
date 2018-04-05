import numpy as np
import os
import gzip
import struct
from urllib.request import urlretrieve

from data_set import DataSet


class MnistLike(DataSet):
    def __init__(self, data_url, name, batch_size, shuffle=True, repeat=True):
        super().__init__(batch_size, shuffle, repeat)
        self.data_url = data_url
        self.path = os.path.join(".data", name)
        self.basenames = {
            "train-images-idx3-ubyte.gz": "train_images",
            "train-labels-idx1-ubyte.gz": "train_labels",
            "t10k-images-idx3-ubyte.gz": "test_images",
            "t10k-labels-idx1-ubyte.gz": "test_labels",
        }

    def test(self):
        return

    def prepare(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        urls = [self.data_url + f for f in self.basenames]
        for url, basename in zip(urls, self.basenames):
            output_file_name = os.path.join(self.path, basename)
            if not os.path.isfile(output_file_name):
                print("downloading file %s..." % basename)
                urlretrieve(url, output_file_name)
        self.load()

    def load(self):
        for file, target in self.basenames.items():
            file_path = os.path.join(self.path, file)
            with gzip.open(file_path) as handle:
                binary_magic = format(struct.unpack(">I", handle.read(8)), '08X')
                assert binary_magic[:4] == "0000"
                assert binary_magic[4:6] == "08"
                dimensions = int(binary_magic[6:], 16)
                shape = struct.unpack(">" + "I" * dimensions, handle.read(8))
                data = np.fromstring(handle.read(), dtype=np.int8)
                setattr(self, target, data.reshape(shape))

    # def __iter__(self):
    #     for x in range(10):
    #         yield x
