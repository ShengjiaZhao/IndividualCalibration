import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from datasets import *
from models import *
from utils import *
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=7)
parser.add_argument('--gpu', type=str, default=3)
parser.add_argument('--dataset', type=str, default='traffic')
parser.add_argument('--coeff', type=float, default=0.2) # When coeff is 0 completely PAI loss, when 1 completely NLL loss
parser.add_argument('--recalibrate', action='store_true')
parser.add_argument('--klcoeff', type=float, default=0.0)
parser.add_argument('--arch', type=str, default='convsmall')
parser.add_argument('--batch_size', type=int, default=128)


args = parser.parse_args()
device = torch.device("cuda:%d" % args.gpu)
args.device = device

architectures = {'fcsmall': FcSmall, 'convsmall': ConvSmall, 'convmid': ConvMid}
datasets = {'crime': CrimeDataset, 'traffic': TrafficDataset, 'toy': ToyDataset}





while True:
    dataset = datasets[args.dataset](device=device)

    # Create logging directory and create tensorboard and text loggers
    log_dir = '/data/calibration/log2/%s/arch=%s-coeff=%.2f-kl=%.2f-recalib=%r-run=%d/' % (args.arch, args.dataset, args.coeff, args.klcoeff, args.recalibrate, args.run)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    logger = open(os.path.join(log_dir, 'result.txt'), 'w')

    # Define model
    model = architectures[args.arch](x_dim=dataset.x_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Variables to determine early stopping
    test_losses = []

    for i in range(1, 50000):
        # Apply standard training step
        model.train()
        optimizer.zero_grad()

        train_x, train_y = dataset.train_batch(args.batch_size)
        cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(train_x, train_y)
        loss = (1 - args.coeff) * loss_cdf + args.coeff * loss_nll

        loss.backward()
        optimizer.step()

        # Log training performance
        if i % 10 == 0:
            writer.add_scalar('cdf_l1', loss_cdf, global_step=i)
            writer.add_scalar('stddev', loss_stddev, global_step=i)
            # writer.add_scalar('l2', loss_l2, global_step=i)
            writer.add_scalar('nll', loss_nll, global_step=i)
            writer.add_scalar('total', loss, global_step=i)

        # Log test performance
        if i % 100 == 0:
            # Computer test loss
            model.eval()
            with torch.no_grad():
                num_iter = min(dataset.test_features.shape[0] // args.batch_size, 10)
                loss_cdf, loss_stddev, loss_nll, loss = [], [], [], []
                for _ in range(num_iter):
                    val_x, val_y = dataset.test_batch(args.batch_size)
                    _, loss_cdf_, loss_stddev_, loss_nll_ = model.eval_all(val_x, val_y)
                    loss_ = (1 - args.coeff) * loss_cdf_ + args.coeff * loss_nll_
                    loss_cdf.append(loss_cdf_.cpu().numpy())
                    loss_stddev.append(loss_stddev_.cpu().numpy())
                    loss_nll.append(loss_nll_.cpu().numpy())
                    loss.append(loss_.cpu().numpy())
                loss_cdf = np.mean(loss_cdf)
                loss_stddev = np.mean(loss_stddev)
                loss_nll = np.mean(loss_nll)
                loss = np.mean(loss)

            writer.add_scalar('cdf_l1_test', loss_cdf, global_step=i)
            writer.add_scalar('stddev_test', loss_stddev, global_step=i)
            writer.add_scalar('nll_test', loss_nll, global_step=i)
            writer.add_scalar('total_test', loss, global_step=i)

            # Early stop if at least 3k iterations and test loss does not improve on average in the past 1k iteration
            test_losses.append(loss)
            if False and len(test_losses) > 30 and np.mean(test_losses[-10:]) >= np.mean(test_losses[-20:-10]):
                # Log the final test loss
                logger.write("%d %f %f %f " % (i, loss, loss_nll, loss_stddev))

                # Compute the recalibration function if necessary
                if args.recalibrate:
                    train_x, train_y = dataset.train_batch()
                    model.recalibrate(train_x, train_y)
                break

            if i % 1000 == 0:
                print("Iteration %d, loss = %.3f" % (i, loss))
                val_x, val_y = dataset.test_batch()
                # Evaluate calibration of subsets
                ratios = [0.2, 1.0]
                for ratio in ratios:
                    ece_s, ece_l = eval_ece(val_x, val_y, model, ratio,
                                            plot_func=lambda image: writer.add_image('calibration %.1f' % ratio, image, i))
                    logger.write("%f " % max(ece_s, ece_l))
                logger.write('\n')

                cost, risk = compute_risk(val_x, val_y, model,
                                          plot_func=lambda image: writer.add_image('trade-off', image, i))
    # Evaluate stddev
    stddev_mc = eval_stddev(val_x, model)
    logger.write("%f\n" % stddev_mc)

    args.run += 1
