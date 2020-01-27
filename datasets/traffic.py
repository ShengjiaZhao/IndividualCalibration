import numpy as np
from matplotlib import pyplot as plt
import os
from datasets.base import *


class TrafficDataset(Dataset):
    def __init__(self, device=torch.device('cpu'), fc=False):
        Dataset.__init__(self, device)

        reader = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/traffic_train.npz'))
        self.train_features, self.train_labels = self.preprocess(reader)
        self.train_c = reader['c']

        # random_perm = np.random.permutation(range(self.train_features.shape[0]))
        # attributes = attributes[, :]

        # self.train_labels += np.random.normal(0, 0.005, self.train_labels.shape)

        reader = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/traffic_test.npz'))
        self.test_features, self.test_labels = self.preprocess(reader)
        self.test_c = reader['c']

        if fc is True:
            self.train_features = np.reshape(self.train_features, [-1, 32*32])
            self.test_features = np.reshape(self.test_features, [-1, 32*32])
            self.x_dim = 32 * 32
        else:
            self.x_dim = 1

    def preprocess(self, reader):
        features = np.expand_dims(reader['x'], axis=1)
        features = self.noisy(features)
        labels = np.expand_dims(reader['y'], axis=1)
        # labels += np.random.normal(0, 0.005, labels.shape)
        return features, labels

    def mono(self, array):
        return array[:, 0:1, :, :] * 0.2989 + array[:, 1:2, :, :] * 0.5870 + array[:, 2:3, :, :] * 0.1140

    def noisy(self, array):
        return np.clip(array + np.random.normal(0, 0.1, array.shape), a_min=0.0, a_max=1.0)

    # If delta is None, brute force search for best delta. Otherwise use 1-delta as the decision threshold
    def compute_risk(self, model, loss_func, thresh, delta=0.05, plot_func=None, val=True):
        test_x_all, test_y_all = self.test_batch()
        if val is True:
            test_x_all = test_x_all[:2000]
            test_y_all = test_y_all[:2000]
            test_c = self.test_c[:2000]
        else:
            test_x_all = test_x_all[2000:]
            test_y_all = test_y_all[2000:]
            test_c = self.test_c[2000:]

        losses = []
        for group in range(16):
            test_x = test_x_all[test_c == group]
            test_y = test_y_all[test_c == group]

            thresh_y = torch.ones_like(test_y) * thresh
            with torch.no_grad():
                cdf1 = model.eval_all(test_x, thresh_y)[0]
                cdf1 = model.apply_recalibrate(cdf1)

                cdf2 = model.eval_all(test_x, 0.85)[0]
                cdf2 = model.apply_recalibrate(cdf2)

            a = 0.5 * (cdf1 < 1. - delta).float() + 0.5 * (cdf2 < 1. - delta).float()
            loss = loss_func(test_y, a, thresh=thresh)
            losses.append(loss.mean().cpu().numpy())

        if plot_func is not None:
            plt.hist(losses, 20)
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            plot_func(image)
            plt.close()
        return losses

    """
    Compute the ECE for natural groups in the dataset
    Input: the model to be evaluated. If group is None then all groups will be evaluated
    Output: eces: a list of value, each value is the ECE error for a group
            names: a list of strings, each string is the name for the group
    """
    def compute_ece_group(self, model, group=None, plot_func=None, val=True):
        test_x, test_y = self.test_batch()
        if val is True:
            test_x = test_x[:2000]
            test_y = test_y[:2000]
            test_c = self.test_c[:2000]
        else:
            test_x = test_x[2000:]
            test_y = test_y[2000:]
            test_c = self.test_c[2000:]

        if group is not None:
            test_x = test_x[test_c == group]
            test_y = test_y[test_c == group]

        with torch.no_grad():
            cdf = model.eval_all(test_x, test_y)[0].cpu().numpy().astype(np.float)
        cdf = model.apply_recalibrate(cdf)[:, 0]
        cdf = np.sort(cdf)

        # Compute calibration error
        err = np.mean(np.abs(cdf - np.linspace(0, 1, cdf.shape[0])))

        # Get calibration curve
        if plot_func is not None:
            plt.figure(figsize=(3, 3))
            plt.plot(cdf, np.linspace(0, 1, cdf.shape[0]))
            plt.plot(np.linspace(0, 1, cdf.shape[0]), np.linspace(0, 1, cdf.shape[0]))
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = PIL.Image.open(buf)
            image = ToTensor()(image)
            plot_func(image)
            plt.close()

        return err

if __name__ == '__main__':
    dataset = TrafficDataset()
    bx, by = dataset.train_batch(100)
    bx, by = dataset.train_batch(100)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(bx[i, 0, :, :], cmap='gray')
        plt.title('%.3f' % by[i])
    plt.show()

    plt.hist(dataset.test_labels.flatten(), bins=20)
    plt.show()

    print(bx.shape)

