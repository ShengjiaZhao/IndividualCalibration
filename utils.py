from datasets import *
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


# Compute the calibration error on the worst subset of samples
# y: an array of y values
# size: a number in (0, 1], the fraction of the validation set to include in the evaluation
""" Given a set of test input (test_x, test_y) and given model compute the ece score 
test_x: a tensor of shape [batch_size, x_dim]
size_ratio: the proportion of samples to select a sub-group
"""
def eval_ece(test_x, test_y, model, size_ratio, plot_func=None):
    size = int(test_x.shape[0] * size_ratio)
    cdf_list = []
    for i in range(200):
        with torch.no_grad():
            cdf = model.eval_all(test_x, test_y)[0]
            cdf_list.append(cdf)

    cdf_avg = torch.cat(cdf_list, axis=1).mean(axis=1)

    # Smallest elements
    elem = torch.argsort(cdf_avg)[:size]
    cdf_s = cdf_list[-1][elem].cpu().numpy()[:, 0]
    cdf_s = np.sort(cdf_s)

    # Compute calibration error
    err_s = np.mean(np.abs(cdf_s - np.linspace(0, 1, cdf_s.shape[0])))

    # Get calibration curve
    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(cdf_s, np.linspace(0, 1, cdf_s.shape[0]))
        plt.plot(np.linspace(0, 1, cdf_s.shape[0]), np.linspace(0, 1, cdf_s.shape[0]))

    # Smallest elements
    elem = torch.argsort(cdf_avg, descending=True)[:size]
    cdf_l = cdf_list[-1][elem].cpu().numpy()[:, 0]
    cdf_l = np.sort(cdf_l)

    # Compute calibration error
    err_l = np.mean(np.abs(cdf_l - np.linspace(0, 1, cdf_l.shape[0])))

    if plot_func is not None:
        plt.plot(cdf_l, np.linspace(0, 1, cdf_l.shape[0]))
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plot_func(image)
        plt.close()

    return err_s, err_l


# Compute the worst ECE if we break the data into two groups based on ranking in top x% of each individual feature
def eval_ece_by_dim(test_x, test_y, model, plot_func=None):
    size = int(test_x.shape[0])
    size_lb = size // 3
    size_ub = size - size_lb
    num_feat = int(test_x.shape[1])
    mid_point = [torch.sort(test_x[:, i])[0][size // 2] for i in range(num_feat)]

    max_err = -1
    worst_cdf = None
    worst_dim = None
    for i in range(num_feat):
        index1 = test_x[:, i] > mid_point[i]
        index2 = test_x[:, i] <= mid_point[i]

        for k, index in enumerate([index1, index2]):
            test_x_part, test_y_part = test_x[index], test_y[index]
            if test_x_part.shape[0] < size_lb or test_x_part.shape[0] > size_ub:
                continue

            with torch.no_grad():
                cdf = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                cdf = np.sort(cdf)

                # Compute calibration error
                err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

                if err > max_err:
                    max_err = err
                    worst_cdf = cdf
                    worst_dim = [i, k]

    # Get calibration curve
    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(worst_cdf, np.linspace(0, 1, worst_cdf.shape[0]))
        plt.plot(np.linspace(0, 1, worst_cdf.shape[0]), np.linspace(0, 1, worst_cdf.shape[0]))
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plot_func(image)
        plt.close()

    return max_err, worst_dim


# Compute the worst ECE if we break the data into two groups based on ranking in top x% of each individual feature
def eval_ece_by_2dim(test_x, test_y, model, plot_func=None):
    size = int(test_x.shape[0])
    size_lb = size // 4
    size_ub = size - size_lb

    num_feat = int(test_x.shape[1])
    mid_point = [torch.sort(test_x[:, i])[0][size // 2] for i in range(num_feat)]

    max_err = -1
    worst_cdf = None
    worst_dim = None
    for i in range(num_feat):
        for j in range(i + 1, num_feat):
            index1 = (test_x[:, i] > mid_point[i]) & (test_x[:, j] > mid_point[j])
            index2 = (test_x[:, i] > mid_point[i]) & (test_x[:, j] <= mid_point[j])
            index3 = (test_x[:, i] <= mid_point[i]) & (test_x[:, j] > mid_point[j])
            index4 = (test_x[:, i] <= mid_point[i]) & (test_x[:, j] <= mid_point[j])

            for k, index in enumerate([index1, index2, index3, index4]):
                test_x_part, test_y_part = test_x[index], test_y[index]
                if test_x_part.shape[0] < size_lb or test_x_part.shape[0] > size_ub:
                    continue

                with torch.no_grad():
                    cdf = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                    cdf = np.sort(cdf)

                    # Compute calibration error
                    err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

                    if err > max_err:
                        # Re-evaluate to prevent over-fitting
                        cdf = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                        cdf = np.sort(cdf)

                        # Compute calibration error
                        err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

                        max_err = err
                        worst_cdf = cdf
                        worst_dim = [i, j, k]

    # Get calibration curve
    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(worst_cdf, np.linspace(0, 1, worst_cdf.shape[0]))
        plt.plot(np.linspace(0, 1, worst_cdf.shape[0]), np.linspace(0, 1, worst_cdf.shape[0]))
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        plot_func(image)
        plt.close()

    return max_err, worst_dim