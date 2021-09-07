"""
The code refers to DeepLearningForFun(https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Oneflow-Python/CycleGAN) by Ldpe2G
and pytorch-CycleGAN-and-pix2pix(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by junyuanz for implementation.
"""
import time
import argparse

from numpy.lib.npyio import load
import numpy as np
from image import ndarray2image, load_image2ndarray
from networks import ResnetGenerator
import os
import cv2
import oneflow as flow


def main(args):
    opt = args

    datasetA = os.listdir(args.datasetA_path)
    datasetB = os.listdir(args.datasetB_path)

    datasetA_num = len(datasetA)
    datasetB_num = len(datasetB)
    print("dataset A size: %d" % datasetA_num)
    print("dataset B size: %d" % datasetB_num)

    netG_A = ResnetGenerator().to("cuda")
    netG_B = ResnetGenerator().to("cuda")

    netG_A.load_state_dict(flow.load(args.netG_A_dir))
    netG_B.load_state_dict(flow.load(args.netG_B_dir))

    print("Begin transforming from A to B")
    for i in range(datasetA_num):
        if i % 10 == 0:
            print("Transformed %d pictures..." % (i))
        imageA = load_image2ndarray(args.datasetA_path + datasetA[i])
        imageA_tensor = flow.Tensor(imageA).to("cuda")
        imageB_fake = netG_A(imageA_tensor)
        imageA = ndarray2image(imageA)
        imageB = ndarray2image(imageB_fake.numpy())
        result = np.concatenate((imageA, imageB), axis=1)
        cv2.imwrite("%s%d.jpg" % (args.fake_B_save_dir, i), result)

    print("Begin transforming from B to A")
    for i in range(datasetB_num):
        if i % 10 == 0:
            print("Transformed %d pictures..." % (i))
        imageB = load_image2ndarray(args.datasetB_path + datasetB[i])
        imageB_tensor = flow.Tensor(imageB).to("cuda")
        imageA_fake = netG_B(imageB_tensor)
        imageB = ndarray2image(imageB)
        imageA = ndarray2image(imageA_fake.numpy())
        result = np.concatenate((imageB, imageA), axis=1)
        cv2.imwrite("%s%d.jpg" % (args.fake_A_save_dir, i), result)

    print("Finished")


def get_parser(parser=None):
    parser = argparse.ArgumentParser("flags for cycle gan")

    parser.add_argument("--datasetA_path", type=str, default="", help="dataset A path")
    parser.add_argument("--datasetB_path", type=str, default="", help="dataset B path")

    # checkpoint
    parser.add_argument(
        "--netG_A_dir", type=str, default=None, help="saving directory of netG_A_dir"
    )
    parser.add_argument(
        "--netG_B_dir", type=str, default=None, help="saving directory of netG_B_dir"
    )

    # save dir
    parser.add_argument(
        "--fake_B_save_dir", default=None, help="directory for generated images"
    )
    parser.add_argument(
        "--fake_A_save_dir", default=None, help="directory for generated images"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
