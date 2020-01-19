from datasets import *
import numpy as np
import torch

func = lambda x: np.sin(np.sum(6. / (1.2 + x), axis=1, keepdims=True))


class ToyDataset(Dataset):
    def __init__(self, x_dim=1):
        Dataset.__init__(self)
        self.x_dim = x_dim
        self.names = ['%d' % i for i in range(x_dim)]

    def train_batch(self, batch_size=256):
        bx = np.random.uniform(-1, 1, (batch_size // 2, self.x_dim))
        bx2 = np.random.uniform(-0.5, 0.5, (batch_size // 2, self.x_dim))
        bx = np.concatenate([bx, bx2], axis=0)
        by = func(bx)
        return bx, by

    def test_batch(self, batch_size=256):
        return self.train_batch(batch_size)


if __name__ == '__main__':
    dataset = ToyDataset()
    bx, by = dataset.train_batch()
    plt.scatter(bx[:, 0], by[:, 0])
    plt.show()

