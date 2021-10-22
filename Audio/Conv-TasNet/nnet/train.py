#!/usr/bin/env python

import argparse
import oneflow as flow
from libs.trainer import SiSnrTrainer
from libs.dataset import make_dataloader
from conv_tas_net import ConvTasNet

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# Task related
parser.add_argument('--train_dir', type=str, default="data/wjs0_2mix/tr/",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--dev_dir', type=str, default="data/wjs0_2mix/cv/",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--fs', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--chunk_len', default=4, type=int,
                    help='Segment length (seconds)')
parser.add_argument('--num_spks', default=2, type=int,
                    help='Number of spearkers')
# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--norm', default='BN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--non_linear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
parser.add_argument('--causal', default=False)
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--min_lr', default=1e-8, type=float,
                    help='weight decay (L2 penalty)')
parser.add_argument('--clip_norm', default=None)
# minibatch
parser.add_argument("--batch-size",
                    type=int,
                    default=8,
                    help="Number of utterances in each batch")
parser.add_argument("--num-workers",
                    type=int,
                    default=0,
                    help="Number of workers used in data loader")
# Training config
parser.add_argument("--epochs",
                    type=int,
                    default=120,
                    help="Number of training epochs")
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=1, type=int,
                    help='Frequency of printing training infomation')


def run(args):
    train_data = {
        "mix_scp":
        args.train_dir + "mix.scp",
        "ref_scp":[args.train_dir + "spk{:d}.scp".format(n)
                   for n in range(1, 1 + args.num_spks)],
        "sample_rate":args.fs,
    }
    dev_data = {
        "mix_scp": args.dev_dir + "mix.scp",
        "ref_scp":[args.dev_dir + "spk{:d}.scp".format(n)
                   for n in range(1, 1 + args.num_spks)],
        "sample_rate": args.fs,
    }
    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   chunk_size=args.chunk_len * args.fs,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 chunk_size=args.chunk_len * args.fs,
                                 num_workers=args.num_workers)
    # network configure
    nnet_conf = {
        "L": args.L,
        "N": args.N,
        "X": args.X,
        "R": args.R,
        "B": args.B,
        "H": args.H,
        "P": args.P,
        "norm": args.norm,
        "num_spks": args.num_spks,
        "non_linear": args.non_linear,
        "causal": args.causal
    }
    # trainer config
    adam_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    trainer_conf = {
        "clip_norm": args.clip_norm,
        "optimizer": args.optimizer,
        "optimizer_kwargs": adam_kwargs,
        "min_lr": args.min_lr,
        "patience": args.patience,
        "factor": args.factor
    }
    nnet = ConvTasNet(**nnet_conf)
    print(nnet)
    device = flow.device("cuda")
    nnet.to(device)
    trainer = SiSnrTrainer(nnet, device, args,**trainer_conf)
    trainer.train(train_loader, dev_loader)

args = parser.parse_args()
run(args)
