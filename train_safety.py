import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numbers, math
import numpy as np
import skimage
from matplotlib import pyplot as plt
import os, math
import io
import PIL.Image
from torchvision.transforms import ToTensor
from datasets import *
from models import *
from utils import *
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alpha', type=float, default=0.2) # When coeff is 0 completely PAI loss, when 1 completely NLL loss
parser.add_argument('--train_size', type=int, default=2000)

args = parser.parse_args()
device = torch.device("cuda:%d" % args.gpu)
args.device = device

# Create logging directory and create tensorboard and text loggers
log_dir = '/data/calibration/safety/alpha=%.2f-train=%d-run=%d' % (args.alpha, args.train_size, args.run)
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

x_dim = 10

def gen_data(batch_size):
    bx = torch.rand(batch_size, x_dim, device=args.device)
    y_pre = bx.sum(axis=1, keepdims=True)
    by = torch.sin(70. / y_pre) * 1.05
    return bx, by

train_x, train_y = gen_data(args.train_size)
test_x, test_y = gen_data(500)

# plt.hist(train_y.cpu().numpy(), bins=50)
# plt.show()
# plt.scatter(train_x.sum(axis=1).cpu(), train_y.cpu())
# plt.show()

model = FcSmall(x_dim=x_dim)
model.to(args.device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Compute risk
def compute_risk(plot_func=None):
    test_x, test_y = gen_data(10000)
    thresh_y = torch.ones_like(test_y)

    with torch.no_grad():
        cdf = model.eval_all(test_x, thresh_y)[0]

    vals = np.linspace(0.9, 1.0, 100)
    cost, risk = [], []
    for val in vals:
        cost.append((cdf < val).float().mean().cpu().numpy())
        risk.append(((cdf >= val).float() * (test_y > 1.0).float()).mean().cpu().numpy())

    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(cost, risk)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plt.tight_layout(pad=3.0)
        plot_func(image)
        plt.close()

    return np.array(cost), np.array(risk)


# Plot the calibration curve
# Compute risk
def calibration_curve(plot_func=None, use_subgroup=False):
    test_x, test_y = gen_data(10000)

    if use_subgroup:
        test_x = test_x[test_y[:, 0] > 1.0]
        test_y = test_y[test_y[:, 0] > 1.0]

    with torch.no_grad():
        cdf = model.eval_all(test_x, test_y)[0].cpu().numpy()
    cdf = np.sort(cdf.flatten())

    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(cdf, np.linspace(0, 1, cdf.shape[0]))
        plt.plot(np.linspace(0, 1, cdf.shape[0]), np.linspace(0, 1, cdf.shape[0]))
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plt.tight_layout()
        plot_func(image)
        plt.close()
    return cost, risk


for i in range(1, 50000):
    # Apply standard training step
    model.train()
    optimizer.zero_grad()

    cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(train_x, train_y)
    loss = (1 - args.alpha) * loss_cdf + args.alpha * loss_nll

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        writer.add_scalar('loss', loss, i)
        with torch.no_grad():
            cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(test_x, test_y)
            test_loss = (1 - args.alpha) * loss_cdf + args.alpha * loss_nll
            writer.add_scalar('test_loss', test_loss, i)

    if i % 100 == 0:
        cost, risk = compute_risk(plot_func=lambda image: writer.add_image('cost-risk', image, i))
        calibration_curve(plot_func=lambda image: writer.add_image('calibration', image, i))
        calibration_curve(plot_func=lambda image: writer.add_image('calibration >1', image, i), use_subgroup=True)
        # Compute optimal cost risk

        risk_weights = [10, 25, 50, 100, 500]
        best_val = [0, 60, 80, 90, 98]
        for (best_val, risk_weight) in zip(best_val, risk_weights):
            utility = cost + risk * risk_weight
            writer.add_scalar('utility %d' % risk_weight, utility[best_val], i)
    #     writer.add_scalar('cost', cost[0], i)
    #     writer.add_scalar('risk', risk, i)
    if i % 1000 == 0:
        print("Iteration %d, loss=%.3f" % (i, loss.detach().cpu()))