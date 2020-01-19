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
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--dataset', type=str, default='crime')
# parser.add_argument('--objective', type=str, default='pai')
parser.add_argument('--coeff', type=float, default=0.1) # When coeff is 0 completely PAI loss, when 1 completely NLL loss
parser.add_argument('--recalibrate', type=bool, default=True)

args = parser.parse_args()
device = torch.device("cuda:%s" % args.gpu)
args.device = device


if args.dataset == 'crime':
    dataset = CrimeDataset()
else:
    dataset = ToyDataset()

log_dir = 'log/%s/coeff=%.2f-recalib=%r-run=%d/' % (args.dataset, args.coeff, args.recalibrate, args.run)
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

writer = SummaryWriter(log_dir)
logger = open(os.path.join(log_dir, 'result.txt'), 'w')

model = FcSmall(x_dim=dataset.x_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()

best_test_loss = 1e+5

early_stop_cnt = 0
prev_loss = None

for i in range(1, 50000):
    model.train()
    optimizer.zero_grad()

    train_x, train_y = dataset.train_batch(128)
    train_x = torch.from_numpy(train_x).float().to(device)
    train_y = torch.from_numpy(train_y).float().to(device)
    cdf, loss_cdf, loss_cdf_kl, loss_stddev, loss_center, loss_nll = model.eval_all(train_x, train_y)
    loss = (1 - args.coeff) * (loss_cdf_kl + loss_cdf * 0.0) + args.coeff * loss_nll

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        writer.add_histogram('cdf', cdf, global_step=i)
        writer.add_scalar('cdf_l1', loss_cdf, global_step=i)
        writer.add_scalar('cdf_kl', loss_cdf_kl, global_step=i)
        writer.add_scalar('stddev', loss_stddev, global_step=i)
        writer.add_scalar('center', loss_center, global_step=i)
        writer.add_scalar('nll', loss_nll, global_step=i)
        writer.add_scalar('total', loss, global_step=i)

    if i % 100 == 0:
        # Computer test loss
        model.eval()
        with torch.no_grad():
            val_x, val_y = dataset.test_batch()
            val_x = torch.from_numpy(val_x).float().to(device)
            val_y = torch.from_numpy(val_y).float().to(device)
            cdf, loss_cdf, loss_cdf_kl, loss_stddev, loss_center, loss_nll = model.eval_all(val_x, val_y)
            loss = (1 - args.coeff) * (loss_cdf_kl * 0.1 + loss_cdf) + args.coeff * loss_nll

        writer.add_histogram('cdf_test', cdf, global_step=i)
        writer.add_scalar('cdf_l1_test', loss_cdf, global_step=i)
        writer.add_scalar('cdf_kl_test', loss_cdf_kl, global_step=i)
        writer.add_scalar('stddev_test', loss_stddev, global_step=i)
        writer.add_scalar('center_test', loss_center, global_step=i)
        writer.add_scalar('nll_test', loss_nll, global_step=i)
        writer.add_scalar('total_test', loss, global_step=i)
        logger.write("%d %f %f %f " % (i, loss, loss_nll, loss_stddev))

        if args.recalibrate is True:
            train_x, train_y = dataset.train_batch()
            train_x = torch.from_numpy(train_x).float().to(device)
            train_y = torch.from_numpy(train_y).float().to(device)
            model.recalibrate(train_x, train_y)

        if i % 100 == 0:
            # ratios = np.linspace(0.01, 1.0, 50)
            # for ratio in ratios:
            #     ece_s, ece_l = eval_ece(val_x, val_y, model, ratio)
            #     logger.write("%f " % max(ece_s, ece_l))
        # else:
            ratios = [0.1, 0.2, 1.0]
            for ratio in ratios:
                ece_s, ece_l = eval_ece(val_x, val_y, model, ratio,
                                        plot_func=lambda image: writer.add_image('calibration %.1f' % ratio, image, i))
                writer.add_scalar('ece_s @ %.1f' % ratio, ece_s, i)
                writer.add_scalar('ece_l @ %.1f' % ratio, ece_l, i)
                logger.write("%f " % max(ece_s, ece_l))

        ece_w, dim = eval_ece_by_dim(val_x, val_y, model,
                                plot_func=lambda image: writer.add_image('calibration-w', image, i))
        writer.add_scalar('ece_w', ece_w, i)
        writer.add_text('worst-w', '%s-%d' % (dataset.names[dim[0]], dim[1]), i)
        logger.write("%f " % ece_w)

        if dataset.x_dim != 1:
            ece_w2, dim = eval_ece_by_2dim(val_x, val_y, model,
                                      plot_func=lambda image: writer.add_image('calibration-w2', image, i))
            writer.add_scalar('ece_w2', ece_w2, i)
            writer.add_text('worst-w2', '%s-%s-%d' % (dataset.names[dim[0]], dataset.names[dim[1]], dim[2]), i)
            logger.write("%f " % ece_w2)

        logger.write('\n')
        logger.flush()
        print("Iteration %d, loss = %.3f" % (i, loss))

        # Early stop if validation loss does not improve by at least 0.01% in 10 consecutive attempts
        if prev_loss is None:
            prev_loss = loss
        elif prev_loss <= loss * 1.0001:
            early_stop_cnt += 1
            if early_stop_cnt == 10:
                break
        else:
            prev_loss = loss
            early_stop_cnt = 0

