import torch
import io
from torchvision.transforms import ToTensor
import PIL.Image

class Dataset:
    def __init__(self, device):
        self.x_dim = [0]

        self.train_ptr = 0
        self.test_ptr = 0

        self.train_features = None
        self.test_features = None
        self.train_labels = None
        self.test_labels = None

        self.device = device

    def train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.train_features.shape[0]
            self.train_ptr = 0
        if self.train_ptr + batch_size > self.train_features.shape[0]:
            self.train_ptr = 0
        bx, by = self.train_features[self.train_ptr:self.train_ptr+batch_size], \
                 self.train_labels[self.train_ptr:self.train_ptr+batch_size]
        self.train_ptr += batch_size
        if self.train_ptr == self.train_features.shape[0]:
            self.train_ptr = 0
        return torch.from_numpy(bx).float().to(self.device), torch.from_numpy(by).float().to(self.device)

    def test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.test_features.shape[0]
            self.test_ptr = 0
        if self.test_ptr + batch_size > self.test_features.shape[0]:
            self.test_ptr = 0
        bx, by = self.test_features[self.test_ptr:self.test_ptr+batch_size], \
                 self.test_labels[self.test_ptr:self.test_ptr+batch_size]
        self.test_ptr += batch_size
        if self.test_ptr == self.test_features.shape[0]:
            self.test_ptr = 0
        return torch.from_numpy(bx).float().to(self.device), torch.from_numpy(by).float().to(self.device)