#!/usr/bin/env python

import os
import argparse
import oneflow as flow
import numpy as np
from conv_tas_net import ConvTasNet
from libs.audio import WaveReader, write_wav

parser = argparse.ArgumentParser(
    description="Command to do speech separation in time domain using ConvTasNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--input", type=str, required=True, help="Script for input waveform")
parser.add_argument(
    "--fs", type=int, default=8000, help="Sample rate for mixture input")
parser.add_argument("--dump-dir",type=str,default="sps_tas",help="Directory to dump separated results out")
# Network architecture
parser.add_argument('--num_spks', default=2, type=int,
                    help='Number of spearkers')
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
parser.add_argument('--model_path', default='exp/temp/final.pth.tar',
                    help='Location to save best validation model')


def separating(args):
    #model
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
    nnet = ConvTasNet(**nnet_conf)
    nnet.load_state_dict(flow.load(args.model_path))
    device = flow.device("cuda")
    nnet.to(device)
    
    with flow.no_grad():
        mix_input = WaveReader(args.input, sample_rate=args.fs)
        for key, mix_samps in mix_input:
            raw = flow.tensor(mix_samps, dtype=flow.float32, device=device)
            sps = nnet(raw)
            spks = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            norm = np.linalg.norm(mix_samps, np.inf)
            for idx, samps in enumerate(spks):
                samps = samps[:mix_samps.size]
                samps = samps * norm / np.max(np.abs(samps))
                write_wav(
                    os.path.join(args.dump_dir, "spk{}/{}".format(
                        idx + 1, key)),
                    samps,
                    fs=args.fs)

args = parser.parse_args()
separating(args)
