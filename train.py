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
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--dataset', type=str, default='crime')
parser.add_argument('--objective', type=str, default='nll')

args = parser.parse_args()
device = torch.device("cuda:%s" % args.gpu)
args.device = device

if args.dataset == 'crime':
    dataset = CrimeDataset()
else:
    dataset = ToyDataset()

log_dir = 'log/%s/objective=%s-run=%d/' % (args.dataset, args.objective, args.run)
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
writer = SummaryWriter(log_dir)

model = FcSmall(x_dim=dataset.x_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()

for i in range(1, 50000):
    optimizer.zero_grad()

    train_x, train_y = dataset.train_batch(128)
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    cdf, loss_cdf, loss_cdf_kl, loss_stddev, loss_center, loss_nll = model.eval_all(train_x, train_y)

    if args.objective == 'pai':
        loss = loss_cdf_kl + loss_cdf * 0.1 + 0.1 * loss_stddev
    else:
        loss = loss_nll

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        writer.add_histogram('cdf', cdf, global_step=i)
        writer.add_scalar('cdf_l1', loss_cdf, global_step=i)
        writer.add_scalar('cdf_kl', loss_cdf_kl, global_step=i)
        writer.add_scalar('stddev', loss_stddev, global_step=i)
        writer.add_scalar('center', loss_center, global_step=i)
        writer.add_scalar('nll', loss_nll, global_step=i)

    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_x, val_y = dataset.test_batch()
            val_x = torch.from_numpy(val_x).float().to(device)
            val_y = torch.from_numpy(val_y).float().to(device)
            cdf, loss_cdf, loss_cdf_kl, loss_stddev, loss_center, loss_nll = model.eval_all(val_x, val_y)
        model.train()

        writer.add_histogram('cdf_test', cdf, global_step=i)
        writer.add_scalar('cdf_l1_test', loss_cdf, global_step=i)
        writer.add_scalar('cdf_kl_test', loss_cdf_kl, global_step=i)
        writer.add_scalar('stddev_test', loss_stddev, global_step=i)
        writer.add_scalar('center_test', loss_center, global_step=i)
        writer.add_scalar('nll_test', loss_nll, global_step=i)

    if i % 1000 == 0:
        ratios = [0.1, 0.2, 0.5, 1.0]
        for ratio in ratios:
            ece_s, ece_l = eval_ece(val_x, val_y, model, ratio,
                                    plot_func=lambda image: writer.add_image('calibration %.1f' % ratio, image, i))
            writer.add_scalar('ece_s @ %.1f' % ratio, ece_s, i)
            writer.add_scalar('ece_l @ %.1f' % ratio, ece_l, i)
        ece_w, dim = eval_ece_by_dim(val_x, val_y, model,
                                plot_func=lambda image: writer.add_image('calibration-w', image, i))
        writer.add_scalar('ece_w', ece_w, i)
        writer.add_text('worst-w', '%s-%d' % (dataset.names[dim[0]], dim[1]))

        if dataset.x_dim != 1:
            ece_w2, dim = eval_ece_by_2dim(val_x, val_y, model,
                                      plot_func=lambda image: writer.add_image('calibration-w2', image, i))
            writer.add_scalar('ece_w2', ece_w, i)
            writer.add_text('worst-w2', '%s-%s-%d' % (dataset.names[dim[0]], dataset.names[dim[1]], dim[2]))

        print("Iteration %d, loss = %.3f" % (i, loss))

