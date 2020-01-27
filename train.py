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
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=600)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='traffic')
parser.add_argument('--coeff', type=float, default=1.0) # When coeff is 0 completely PAI loss, when 1 completely NLL loss
parser.add_argument('--recalibrate', action='store_true')
parser.add_argument('--klcoeff', type=float, default=0.0)
parser.add_argument('--arch', type=str, default='convsmall')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--loss_eps', type=float, default=0.02)
parser.add_argument('--thresh', type=float, default=0.7)
parser.add_argument('--delta', type=float, default=0.05)

args = parser.parse_args()
device = torch.device("cuda:%d" % args.gpu)
args.device = device

architectures = {'fcsmall': FcSmall, 'convsmall': ConvSmall, 'convmid': ConvMid}
datasets = {'crime': CrimeDataset, 'traffic': TrafficDataset, 'toy': ToyDataset}


# y should be a torch tensor of shape (batch_size, 1)
# a should be a torch tensor of shape (batch_size, 1)


# def decision_loss(y, a, eps=0.01, thresh=0.8):
#     return (y < thresh).float() * a * (thresh - y) + \
#         (y >= thresh).float() * (1/(a ** 2 + eps) - 1/(1 + eps))


def eval_extra(model, writer, logger):
    model.eval()
    val_x, val_y = dataset.test_batch()
    if args.recalibrate:
        model.recalibrate(val_x, val_y)

    for delta in [0.005, 0.007, 0.01, 0.02]:
        for loss_eps in [0.01, 0.02, 0.1, 0.5]:
            def decision_loss(y, a, thresh):
                return (y < thresh).float() * a * (thresh - y) + \
                       (y >= thresh).float() * (((y - thresh) / (1 - thresh)) / (a ** 2 + loss_eps) - (
                        (y - thresh) / (1 - thresh)) / (1 + loss_eps))
            dloss = dataset.compute_risk(model, loss_func=decision_loss, thresh=args.thresh, delta=delta,
                                         plot_func=lambda image: writer.add_image('loss_hist', image, i))
            # writer.add_scalar('dloss', np.mean(dloss), i)
            logger.write('dloss ')
            for loss_val in dloss:
                logger.write('%f ' % loss_val)
            logger.write('\n')

    # Subgroup ECE
    logger.write('ece ')
    for g in range(16):
        dece = dataset.compute_ece_group(model, g,
                                         plot_func=lambda image: writer.add_image('calib curve %d' % g, image, i))
        writer.add_scalar('ece group %d' % g, dece, i)
        logger.write('%f ' % dece)
    dece = dataset.compute_ece_group(model,
                                     plot_func=lambda image: writer.add_image('calib curve all', image, i))
    writer.add_scalar('ece all', dece, i)
    logger.write('%f ' % dece)
    logger.write('\n')

    val_x, val_y = dataset.test_batch()

    # Worst case ECE
    logger.write('worst-ece ')
    ratios = [0.2, 1.0]
    for ratio in ratios:
        ece_s, ece_l = eval_ece(val_x, val_y, model, ratio, resolution=2000,
                                plot_func=lambda image: writer.add_image('calibration %.1f' % ratio, image, i))
        writer.add_scalar('ece @ %.1f' % ratio, max(ece_s, ece_l), i)
        logger.write('%f ' % max(ece_s, ece_l))
    logger.write('\n')

    # Evaluate stddev
    stddev_mc = eval_stddev(val_x, model)
    logger.write("stddev %f\n" % stddev_mc)

start_time = time.time()
for rep in range(200):
    dataset = datasets[args.dataset](device=device)

    # Create logging directory and create tensorboard and text loggers
    log_dir = '/data/calibration/log3/%s/arch=%s-thresh=%.2f-delta=%.2f-coeff=%.2f-kl=%.2f-recalib=%r-losseps=%.3f-run=%d/' \
              % (args.dataset, args.arch, args.thresh, args.delta, args.coeff, args.klcoeff, args.recalibrate, args.loss_eps, args.run)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    logger = open(os.path.join(log_dir, 'result.txt'), 'w')

    # Define model
    model = architectures[args.arch](x_dim=dataset.x_dim, args=args)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Variables to determine early stopping
    test_losses = []

    for i in range(1, 20000):
        # Apply standard training step
        model.train()
        optimizer.zero_grad()

        train_x, train_y = dataset.train_batch(args.batch_size)
        cdf, loss_cdf, loss_stddev, loss_nll = model.eval_all(train_x, train_y)
        loss = (1 - args.coeff) * loss_cdf + args.coeff * loss_nll

        loss.backward()
        optimizer.step()

        # Log training performance
        if i % 1000 == 0:
            print("Run %d Iteration %d, loss = %.3f, time elapsed=%.1f" % (args.run, i, loss, time.time() - start_time))
            start_time = time.time()

        # if i % 10 == 0:
        #     writer.add_scalar('cdf_l1', loss_cdf, global_step=i)
        #     writer.add_scalar('stddev', loss_stddev, global_step=i)
        #     writer.add_scalar('nll', loss_nll, global_step=i)
        #     writer.add_scalar('total', loss, global_step=i)

        if i == 15000:
            eval_extra(model, writer, logger)
        # Log test performance
        # if i % 100 == 0:
            # Computer test loss
            # model.eval()
            # with torch.no_grad():
            #     val_x, val_y = dataset.test_batch()
            #     _, loss_cdf, loss_stddev, loss_nll = model.eval_all(val_x, val_y)
            #     loss = (1 - args.coeff) * loss_cdf + args.coeff * loss_nll
            #
            # writer.add_scalar('cdf_l1_test', loss_cdf, global_step=i)
            # writer.add_scalar('stddev_test', loss_stddev, global_step=i)
            # writer.add_scalar('nll_test', loss_nll, global_step=i)
            # writer.add_scalar('total_test', loss, global_step=i)
            #
            # # Early stop if at least 3k iterations and test loss does not improve on average in the past 1k iteration
            # test_losses.append(loss.cpu().numpy())
            # if False and len(test_losses) > 20 and np.mean(test_losses[-10:]) >= np.mean(test_losses[-20:-10]):
            #     # Log the final test loss
            #     logger.write("%d %f %f %f " % (i, loss, loss_nll, loss_stddev))
            #
            #     val_x, val_y = dataset.test_batch()
            #     dloss, dthresh = compute_risk(val_x, val_y, model, loss_func=decision_loss, thresh=args.thresh)
            #     logger.write("%f %f " % (dloss, dthresh))
            #
            #     # Compute the recalibration function if necessary
            #     if args.recalibrate:
            #         train_x, train_y = dataset.train_batch(2000)
            #         model.recalibrate(train_x, train_y)
            #     break

            # dloss, dthresh = compute_risk(val_x, val_y, model, loss_func=decision_loss, delta=args.delta, thresh=args.thresh,
            #                               plot_func=lambda image: writer.add_image('decision_curve', image, i))
            # writer.add_scalar('decision_loss', dloss, i)
            # logger.write("%f " % (dloss))

            # dloss, dthresh = compute_risk(val_x, val_y, model, loss_func=decision_loss, delta=0.03, thresh=args.thresh)
            # logger.write("%f " % (dloss))

            # dloss, dthresh = compute_risk(val_x, val_y, model, loss_func=decision_loss, thresh=args.thresh)
            # writer.add_scalar('decision_loss_min', dloss, i)
            # writer.add_scalar('decision_thresh_min', dthresh, i)
            # logger.write("%f %f " % (dloss, dthresh))

            # train_x, train_y = dataset.train_batch()
            # dloss, dthresh = compute_risk(train_x, train_y, model, loss_func=decision_loss, thresh=args.thresh)
            # writer.add_scalar('decision_loss_train', dloss, i)
            # writer.add_scalar('decision_thresh_train', dthresh, i)
            # logger.write("%f %f\n" % (dloss, dthresh))

            # # Compute the worst case ECE score
    eval_extra(model, writer, logger)
    args.run += 1
    logger.close()
    # print("Running run number %d" % args.run)
    continue



    # Evaluate calibration of subsets
    ratios = np.linspace(0.01, 1.0, 50)
    for ratio in ratios:
        ece_s, ece_l = eval_ece(val_x, val_y, model, ratio)
        logger.write("%f " % max(ece_s, ece_l))
    logger.write('\n')

    # Evaluate calibration of identifiable subgroups
    ece_w, dim = eval_ece_by_dim(val_x, val_y, model,
                                 plot_func=lambda image: writer.add_image('calibration-w', image))
    logger.write('%f %s-%d\n' % (ece_w, dataset.names[dim[0]], dim[1]))

    if dataset.x_dim != 1:
        ece_w2, dim = eval_ece_by_2dim(val_x, val_y, model,
                                       plot_func=lambda image: writer.add_image('calibration-w2', image))
        # writer.add_scalar('ece_w2', ece_w2, i)
        # writer.add_text('worst-w2', '%s-%s-%d' % (dataset.names[dim[0]], dataset.names[dim[1]], dim[2]), i)
        logger.write("%f %s-%s-%d\n" % (ece_w2, dataset.names[dim[0]], dataset.names[dim[1]], dim[2]))
    logger.close()


