import numpy as np
from matplotlib import pyplot as plt
import os
from datasets.base import *
import torch


class CrimeDataset(Dataset):
    def __init__(self, device=torch.device('cpu')):
        Dataset.__init__(self, device)

        reader = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/communities.data'))

        attributes = []
        while True:
            line = reader.readline().split(',')
            if len(line) < 128:
                break
            line = ['-1' if val == '?' else val for val in line]
            line = np.array(line[5:], dtype=np.float)
            attributes.append(line)
        reader.close()

        attributes = np.stack(attributes, axis=0)

        reader = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/names'))
        names = []
        for i in range(128):
            line = reader.readline().split()[1]
            if i >= 5:
                names.append(line)
        names = np.array(names)

        attributes = attributes[np.random.permutation(range(attributes.shape[0])), :]

        val_size = 500
        self.train_labels = attributes[val_size:, -1:]
        self.test_labels = attributes[:val_size:, -1:]

        attributes = attributes[:, :-1]
        selected = np.argwhere(np.array([np.min(attributes[:, i]) for i in range(attributes.shape[1])]) >= 0).flatten()
        self.train_features = attributes[val_size:, selected]
        self.test_features = attributes[:val_size:, selected]
        self.names = names[selected]

        self.x_dim = self.train_features.shape[1]

    # Compute the worst ECE if we break the data into two groups based on ranking in top x% of each individual feature
    def eval_ece_by_dim(self, test_x, test_y, model, plot_func=None):
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



if __name__ == '__main__':
    dataset = CrimeDataset()
    print(dataset.names)
    print(dataset.train_features.shape, dataset.train_labels.shape)

    plt.hist(dataset.test_labels, bins=20)
    plt.show()


