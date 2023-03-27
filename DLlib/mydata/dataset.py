import numpy as np
import torchvision


class NdarrayDataset(object):
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def MNIST(root, train=True):
    mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)
    images = mnist.data.numpy() / 255  # N*28*28
    labels = mnist.targets.numpy()  # N,
    images = np.expand_dims(images, axis=1)  # N*1*18*28
    return NdarrayDataset(images, labels)


def FashionMNIST(root, train=True):
    fashionmnist = torchvision.datasets.FashionMNIST(
        root=root, train=train, download=True
    )
    images = fashionmnist.data.numpy() / 255  # N*28*28
    labels = fashionmnist.targets.numpy()  # N,
    images = np.expand_dims(images, axis=1)  # N*1*18*28
    return NdarrayDataset(images, labels)
