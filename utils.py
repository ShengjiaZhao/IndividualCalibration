from datasets import *
import torch
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d


# Compute decision risk
# If delta is None, brute force search for best delta. Otherwise use 1-delta as the decision threshold
def compute_risk(test_x, test_y, model, loss_func, thresh, delta=None, plot_func=None):
    thresh_y = torch.ones_like(test_y) * thresh

    with torch.no_grad():
        cdf = model.eval_all(test_x, thresh_y)[0]

    if delta is None:
        vals = np.linspace(0.9, 1.0, 100)
        losses = []
        for val in vals:
            a = (cdf < val).float()
            loss = loss_func(test_y, a)
            losses.append(loss.mean().cpu())
        return np.min(np.array(losses)), 1-vals[np.argmin(np.array(losses))]
    else:
        a = (cdf < 1. - delta).float()
        loss = loss_func(test_y, a)
        if plot_func is not None:
            num_bins = 10
            bins = np.linspace(0, 1, num_bins+1)
            losses = []
            for i in range(num_bins):
                losses.append(float(loss[(test_y > bins[i]) & (test_y <= bins[i+1])].mean().cpu().numpy()))
                if np.isnan(losses[-1]):
                    losses[-1] = 0.0
            plt.plot(bins[:-1], losses)
            # plt.scatter(test_y.cpu().numpy().flatten(), loss.cpu().numpy().flatten(), alpha=0.2)
            # plt.yscale('log')
            # plt.ylim([0.001, 100])
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            plot_func(image)
            plt.close()
        return float(loss.mean().cpu()), delta


# Compute the calibration error on the worst subset of samples
# y: an array of y values
# size: a number in (0, 1], the fraction of the validation set to include in the evaluation
""" Given a set of test input (test_x, test_y) and given model compute the ece score 
test_x: a tensor of shape [batch_size, x_dim]
size_ratio: the proportion of samples to select a sub-group
"""
def eval_ece(test_x, test_y, model, size_ratio, resolution=2000, plot_func=None):
    size = int(test_x.shape[0] * size_ratio)
    cdf_list = []
    for i in range(resolution):
        with torch.no_grad():
            cdf = model.eval_all(test_x, test_y)[0].cpu().numpy().astype(np.float)
            cdf = model.apply_recalibrate(cdf)
            cdf_list.append(cdf)

    cdf_avg = np.mean(np.concatenate(cdf_list, axis=1), axis=1)

    # Smallest elements
    elem = np.argsort(cdf_avg)[:size]
    cdf_s = cdf_list[-1][elem][:, 0]
    cdf_s = np.sort(cdf_s)

    # Compute calibration error
    err_s = np.mean(np.abs(cdf_s - np.linspace(0, 1, cdf_s.shape[0])))

    # Get calibration curve
    if plot_func is not None:
        plt.figure(figsize=(3, 3))
        plt.plot(cdf_s, np.linspace(0, 1, cdf_s.shape[0]))
        plt.plot(np.linspace(0, 1, cdf_s.shape[0]), np.linspace(0, 1, cdf_s.shape[0]))

    # Smallest elements
    elem = np.argsort(cdf_avg)[-size:]
    cdf_l = cdf_list[-1][elem][:, 0]
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


# Evaluate the standard deviation of the predicted distributions with Monte Carlo
def eval_stddev(test_x, model):
    with torch.no_grad():
        mean_batch, stddev_batch = model(test_x)
        mean_batch = mean_batch.cpu().numpy().astype(np.float)
        stddev_batch = stddev_batch.cpu().numpy().astype(np.float)

    # if model.iso_transform is None:
    #     return np.mean(stddev_batch)
    #
    stddevs = []
    for mean, stddev in zip(mean_batch, stddev_batch):
        # Compute the CDF (represent it by its values between [x-10stddev, x+10stddev])
        # This code could fail if we sample beyond 10x stddev, but that is so unlikely
        # Also we use 10000 samples, but I'm sure the error is negligible
        xs = np.linspace(mean - 10 * stddev, mean + 10 * stddev, 10000)
        ys = norm.cdf((xs - mean) / stddev)
        ys_new = model.apply_recalibrate(ys)

        # Compute approximate inverse CDF
        f = interp1d(ys_new, xs, kind='linear')

        # Draw samples from the distribution by inverse CDF trick
        r = np.random.random(size=(10000,))
        fr = f(r)

        # Compute empirical stddev
        stddevs.append(np.std(fr))
    return np.mean(stddevs)


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
                cdf = model.apply_recalibrate(cdf)
                cdf = np.sort(cdf)

                # Compute calibration error
                err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

                if err > max_err:
                    cdf = model.eval_all(test_x_part, test_y_part)[0].cpu().numpy()[:, 0]
                    cdf = np.sort(cdf)

                    # Compute calibration error
                    err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

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
