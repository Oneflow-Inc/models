from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
import model_search
import datasets
from functions import train, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
import oneflow as flow
import os
import numpy as np
import oneflow.nn as nn
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import random

def main():
    args = cfg.parse_args()

    if args.seed is not None:
        flow.manual_seed(args.random_seed)
        # flow.cuda.manual_seed(args.random_seed)
        # flow.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        flow.backends.cudnn.benchmark = False
        flow.backends.cudnn.deterministic = True
    ngpus_per_node = flow.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)
    args.world_size = flow.env.get_world_size()

def main_worker(gpu, ngpus_per_node, args):
    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # import network
    gen_net = eval('model_search.' + args.gen_model + '.Generator')(args=args)
    dis_net = eval('model_search.' + args.dis_model + '.Discriminator')(args=args)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    gen_net.cuda()
    dis_net.cuda()
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.dis_batch_size = int(args.dis_batch_size / ngpus_per_node)
    args.gen_batch_size = int(args.gen_batch_size / ngpus_per_node)
    args.batch_size = args.dis_batch_size

    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    # gen_net = flow.nn.parallel.DistributedDataParallel(gen_net, broadcast_buffers=False)
    # dis_net = flow.nn.parallel.DistributedDataParallel(dis_net, broadcast_buffers=False)

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = flow.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                         args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = flow.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = flow.optim.AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
        dis_optimizer = flow.optim.AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                              args.g_lr, weight_decay=args.wd)
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = '/home/shikaijie/transgan_of/fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.fid_stat is not None:
        fid_stat = args.fid_stat
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    dataset = datasets.ImageDataset(args, cur_img_size=8)
    train_loader = dataset.train
    train_sampler = dataset.train_sampler
    print(len(train_loader))
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = flow.tensor(np.random.normal(0, 1, (100, args.latent_dim))).cuda()
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = flow.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']

        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])

        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        fixed_z = checkpoint['fixed_z']
        args.path_helper = checkpoint['path_helper']
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        # writer = SummaryWriter(args.path_helper['log_path']) if args.rank == 0 else None
        del checkpoint
    else:
        # create new log dir
        assert args.exp_name
        if args.rank == 0:
            # args.path_helper = set_log_dir('logs', args.exp_name)
            # logger = create_logger(args.path_helper['log_path'])
            # writer = SummaryWriter(args.path_helper['log_path'])
            pass

    # if args.rank == 0:
    #     logger.info(args)
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        train_sampler.set_epoch(epoch)
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage)) if args.rank == 0 else 0
        print(f"path: {args.path_helper['prefix']}") if args.rank == 0 else 0
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              fixed_z,
              lr_schedulers)

        if args.rank == 0 and args.show:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            save_samples(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            load_params(gen_net, backup_param, args)

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch) - 1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param, args, mode="cpu")
            inception_score, fid_score = validate(args, fixed_z, fid_stat, epoch, gen_net, writer_dict)
            # if args.rank == 0:
            #     logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
            load_params(gen_net, backup_param, args)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_model': args.gen_model,
                'dis_model': args.dis_model,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': best_fid,
                'path_helper': args.path_helper,
                'fixed_z': fixed_z
            }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint'):
    flow.save(states, os.path.join(output_dir, filename))
    if is_best:
        flow.save(states, os.path.join(output_dir, 'checkpoint_best'))

if __name__ == '__main__':
    main()

