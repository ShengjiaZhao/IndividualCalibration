import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.isotonic import IsotonicRegression
import numpy as np


class Forecaster(nn.Module):
    def __init__(self, args):
        super(Forecaster, self).__init__()
        self.args = args

    def eval_all(self, bx, by):
        br = torch.rand(bx.shape[0], 1, device=bx.device)
        mean, stddev = self.forward(bx=bx, br=br)
        cdf = 0.5 * (1.0 + torch.erf((by - mean) / stddev / math.sqrt(2)))

        loss_cdf = torch.abs(cdf - br).mean()

        eps = 1e-5
        loss_cdf_kl = cdf * (torch.log(cdf + eps) - torch.log(br + eps)) + \
                      (1 - cdf) * (torch.log(1 - cdf + eps) - torch.log(1 - br + eps))
        loss_cdf_kl = loss_cdf_kl.mean()

        loss_stddev = stddev.mean()

        # loss_l2 = ((by - mean) ** 2).mean()

        # Log likelihood of by under the predicted Gaussian distribution
        loss_nll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((by - mean) / stddev) ** 2 / 2.0)
        loss_nll = loss_nll.mean()

        return cdf, loss_cdf * (1 - self.args.klcoeff) + loss_cdf_kl * self.args.klcoeff, loss_stddev, loss_nll

    def eval_in_batch(self, bx, by, batch_size):
        pass

    def recalibrate(self, bx, by):
        with torch.no_grad():
            cdf = self.eval_all(bx, by)[0].cpu().numpy()[:, 0].astype(np.float)

        cdf = np.sort(cdf)
        lin = np.linspace(0, 1, int(cdf.shape[0]))

        # Insert an extra 0 and 1 to ensure the range is always [0, 1], and trim CDF for numerical stability
        cdf = np.clip(cdf, a_max=1.0-1e-6, a_min=1e-6)
        cdf = np.insert(np.insert(cdf, -1, 1), 0, 0)
        lin = np.insert(np.insert(lin, -1, 1), 0, 0)

        self.iso_transform = IsotonicRegression()
        self.iso_transform.fit_transform(cdf, lin)

    def apply_recalibrate(self, cdf):
        if self.iso_transform is not None:
            # If input tensor output tensor
            # If input numpy array output numpy array
            is_torch = False
            if isinstance(cdf, type(torch.zeros(1))):
                device = cdf.get_device()
                cdf = cdf.cpu().numpy()
                is_torch = True

            original_shape = cdf.shape
            new_cdf = np.reshape(self.iso_transform.transform(cdf.flatten()), original_shape)
            if is_torch:
                new_cdf = torch.from_numpy(new_cdf).to(device)
            return new_cdf
        else:
            return cdf


class FcSmall(Forecaster):
    def __init__(self, args, x_dim=1):
        super(FcSmall, self).__init__(args)
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100 + 1, 100)
        self.fc3 = nn.Linear(100 + 1, 100)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(100, 2)
        self.iso_transform = None

    def forward(self, bx, br=None):
        if br is None:
            br = torch.rand(bx.shape[0], 1, device=bx.device)
        h = F.leaky_relu(self.fc1(bx))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc2(h))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc3(h))
        h = self.drop3(h)
        h = self.fc4(h)
        mean = h[:, 0:1]
        stddev = torch.abs(h[:, 1:2]) + 0.01
        return mean, stddev


class ConvSmall(Forecaster):
    def __init__(self, args, x_dim):
        super(ConvSmall, self).__init__(args)
        self.conv1 = nn.Conv2d(x_dim, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2)
        self.fc1 = nn.Linear(6*6*16+1, 64)
        self.fc2 = nn.Linear(64+1, 64)
        # self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)
        self.iso_transform = None

    def forward(self, bx, br=None):
        if br is None:
            br = torch.rand(bx.shape[0], 1, device=bx.device)
        h = F.leaky_relu(self.conv1(bx))
        # h should be 15x15x32
        h = F.leaky_relu(self.conv2(h))
        # h should be 7x7x32
        h = h.view(bx.shape[0], -1)
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc1(h))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc2(h))
        # h = self.drop2(h)
        h = self.fc3(h)
        mean = h[:, 0:1]
        stddev = torch.abs(h[:, 1:2]) + 0.01
        return mean, stddev


class ConvMid(Forecaster):
    def __init__(self, x_dim):
        super(ConvMid, self).__init__()
        self.conv1 = nn.Conv2d(x_dim, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=2)
        self.fc1 = nn.Linear(6*6*32+1, 100)
        self.fc2 = nn.Linear(100+1, 100)
        # self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 2)
        self.iso_transform = None

    def forward(self, bx, br=None):
        if br is None:
            br = torch.rand(bx.shape[0], 1, device=bx.device)
        h = F.leaky_relu(self.conv1(bx))
        # h should be 15x15x32
        h = F.leaky_relu(self.conv2(h))
        # h should be 7x7x32
        h = h.view(bx.shape[0], -1)
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc1(h))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc2(h))
        # h = self.drop2(h)
        h = self.fc3(h)
        mean = h[:, 0:1]
        stddev = torch.abs(h[:, 1:2]) + 0.01
        return mean, stddev
