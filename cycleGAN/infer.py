"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import argparse

from numpy.lib.npyio import load
import numpy as np

from cycleGAN import CycleGANModel
from image import ImagePool, ndarray2image, load_image2ndarray
from networks import ResnetGenerator
import os
import torch
import re
import random
import cv2
import oneflow.experimental as flow
import oneflow.experimental.nn as nn

def main(args):
    random.seed(1)
    flow.env.init()
    flow.enable_eager_execution()
    opt = args

    datasetA = os.listdir(args.datasetA_path)
    datasetB = os.listdir(args.datasetB_path)

    datasetA_num = len(datasetA)
    datasetB_num = len(datasetB)
    print("dataset A size: %d" % datasetA_num)
    print("dataset B size: %d" % datasetB_num) 

    train_iters = min(datasetA_num, datasetB_num)          
    # parameters = torch.load("./checkpoints/latest_net_G.pth")

    # new_parameters = dict()
    # for key,value in parameters.items():
    #     if "num_batches_tracked" not in key and not key.endswith("running_mean") and not key.endswith("running_var"):
    #         val = value.detach().cpu().numpy()
    #         new_parameters[key] = val

    # htz_model.load_state_dict(new_parameters)
    # flow.save(htz_model.state_dict(), "zth_model_oneflow")
    netG_A = ResnetGenerator(3, 3).to("cuda")
    netG_B = ResnetGenerator(3, 3).to("cuda")

    netG_A.load_state_dict(flow.load(args.netG_A_dir))
    netG_B.load_state_dict(flow.load(args.netG_B_dir))

    print("Begin transforming from A to B")
    for i in range(datasetA_num):
        if i % 10 == 0:
            print ("Transformed %d pictures..." % (i))
        imageA = load_image2ndarray(args.datasetA_path + datasetA[i])
        imageA_tensor = flow.Tensor(imageA).to("cuda")
        imageB_fake = netG_A(imageA_tensor)
        imageA = ndarray2image(imageA)
        imageB = ndarray2image(imageB_fake.numpy())
        result = np.concatenate((imageA, imageB), axis=1)
        cv2.imwrite("%s%d.jpg" % (args.fake_B_save_dir, i), result)

    print("Begin transforming from B to A")
    for i in range(datasetA_num):
        if i % 10 == 0:
            print ("Transformed %d pictures..." % (i))
        imageB = load_image2ndarray(args.datasetB_path + datasetB[i])
        imageB_tensor = flow.Tensor(imageB).to("cuda")
        imageA_fake = netG_B(imageB_tensor)
        imageB = ndarray2image(imageB)
        imageA = ndarray2image(imageA_fake.numpy())
        result = np.concatenate((imageB, imageA), axis=1)
        cv2.imwrite("%s%d.jpg" % (args.fake_A_save_dir, i), result)

    print("Finished")

def get_parser(parser = None):
    parser = argparse.ArgumentParser("flags for cycle gan")

    parser.add_argument("--datasetA_path", type = str, default = "", help = "dataset A path")
    parser.add_argument("--datasetB_path", type = str, default = "", help = "dataset B path")

    # checkpoint
    parser.add_argument("--netG_A_dir", type = str, default = None, help = "saving directory of netG_A_dir")
    parser.add_argument("--netG_B_dir", type = str, default = None, help = "saving directory of netG_B_dir")

    # save dir
    parser.add_argument("--fake_B_save_dir", default = None, help = "directory for generated images")
    parser.add_argument("--fake_A_save_dir", default = None, help = "directory for generated images")

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
