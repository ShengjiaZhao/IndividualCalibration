import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.isotonic import IsotonicRegression
import numpy as np


class FcSmall(nn.Module):
    def __init__(self, x_dim=1):
        super(FcSmall, self).__init__()
        self.fc1 = nn.Linear(x_dim, 100)
        self.fc2 = nn.Linear(100 + 1, 100)
        self.fc3 = nn.Linear(100 + 1, 100)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(100, 2)
        self.iso_transform = None

    def forward(self, bx, br):
        h = F.leaky_relu(self.fc1(bx))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc2(h))
        h = torch.cat([h, br], dim=1)
        h = F.leaky_relu(self.fc3(h))
        h = self.drop3(h)
        h = self.fc4(h)
        mean = h[:, 0:1]
        stddev = torch.sigmoid(h[:, 1:2]) * 10.0 + 0.01
        return mean, stddev

    def eval_all(self, bx, by):
        eps = 1e-5
        br = torch.rand(bx.shape[0], 1, device=bx.device)
        mean, stddev = self.forward(bx, br)
        cdf = 0.5 * (1.0 + torch.erf((by - mean) / stddev / math.sqrt(2)))

        loss_cdf = torch.abs(cdf - br).mean()
        loss_cdf_kl = cdf * (torch.log(cdf + eps) - torch.log(br + eps)) + \
                      (1 - cdf) * (torch.log(1 - cdf + eps) - torch.log(1 - br + eps))
        loss_cdf_kl = loss_cdf_kl.mean()
        loss_stddev = stddev.mean()
        loss_center = torch.abs(mean - by).mean()

        # Log likelihood of by under the predicted Gaussian distribution
        loss_ll = torch.log(stddev) + math.log(2 * math.pi) / 2.0 + (((by - mean) / stddev) ** 2 / 2.0)
        loss_ll = loss_ll.mean()

        return cdf, loss_cdf, loss_cdf_kl, loss_stddev, loss_center, loss_ll

    def recalibrate(self, bx, by):
        with torch.no_grad():
            cdf = self.eval_all(bx, by)[0].cpu().numpy()[:, 0].astype(np.float)

        cdf = np.sort(cdf)

        self.iso_transform = IsotonicRegression()
        self.iso_transform.fit_transform(cdf, np.linspace(0, 1, int(cdf.shape[0])))

    def apply_recalibrate(self, cdf):
        if self.iso_transform is not None:
            original_shape = cdf.shape
            return np.reshape(self.iso_transform.transform(cdf.flatten()), original_shape)
        else:
            return cdf

