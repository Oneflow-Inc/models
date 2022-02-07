from cmath import nan
import os
import oneflow
import oneflow.optim as optim
import argparse
import oneflow.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import numpy as np
from oneflow.nn.parallel import DistributedDataParallel as ddp


def train(args):
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50

    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    training_dataset = args.training_dataset
    save_folder = args.save_folder

    net = RetinaFace(cfg=cfg)
    # print("Printing net...")
    # print(net)

    if args.resume_net is not None:
        print('Loading resume network...')
        state_dict = oneflow.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    # state_dict = oneflow.load("./weights/Resnet50_Final_1")
    # net.load_state_dict(state_dict)
    net = net.to("cuda")

    optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                          momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with oneflow.no_grad():
        priors = priorbox.forward()
        priors = priors.to("cuda")

    net = ddp(net)
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / (batch_size *
                           oneflow.env.get_world_size()))
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    rank = oneflow.env.get_rank()
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(
                dataset, batch_size, shuffle=True, num_workers=1, collate_fn=detection_collate))
            if rank == 0:
                if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                    oneflow.save(net.state_dict(), save_folder +
                             cfg['name'] + '_epoch_' + str(epoch) + '')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, gamma, epoch, step_index, iteration, epoch_size,initial_lr)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to("cuda")
        targets = [anno.to("cuda") for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if rank ==0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                      epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
    if rank == 0:
        oneflow.save(net.state_dict(), save_folder + cfg['name'] + '_Final')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size,initial_lr):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset',
                        default='./data/widerface/train/label.txt', help='Training dataset directory')
    parser.add_argument('--network', default='mobile0.25',
                        help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3,
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None,
                        help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int,
                        help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weight/',
                        help='Location to save checkpoint models')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    train(args)
