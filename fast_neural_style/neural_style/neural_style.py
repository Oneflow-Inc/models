import argparse
import os
import sys
import time
import re
from typing import DefaultDict
import cv2
import random

import numpy as np
import utils
import oneflow as flow
from oneflow.optim import Adam

from utils import load_image, recover_image, normalize_batch, load_image_eval
from transformer_net import TransformerNet
from vgg import vgg16, VGG_WITH_FEATURES, vgg19


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (
            os.path.exists(args.checkpoint_model_dir)
        ):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = "cuda"
    np.random.seed(args.seed)
    # load path of train images
    train_images = os.listdir(args.dataset)
    train_images = [image for image in train_images if not image.endswith("txt")]
    random.shuffle(train_images)
    images_num = len(train_images)
    print("dataset size: %d" % images_num)
    # Initialize transforemer net, optimizer, and loss function
    transformer = TransformerNet().to("cuda")

    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = flow.nn.MSELoss()

    if args.load_checkpoint_dir is not None:
        state_dict = flow.load(args.load_checkpoint_dir)
        transformer.load_state_dict(state_dict)
        print("successfully load checkpoint from " + args.load_checkpoint_dir)

    # load pretrained vgg16
    if args.vgg == "vgg19":
        vgg = vgg19(pretrained=True)
    else:
        vgg = vgg16(pretrained=True)
    vgg = VGG_WITH_FEATURES(vgg.features, requires_grad=False)
    vgg.to("cuda")

    style_image = utils.load_image(args.style_image)
    style_image_recover = recover_image(style_image)
    features_style = vgg(utils.normalize_batch(flow.Tensor(style_image).to("cuda")))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.0
        agg_style_loss = 0.0
        count = 0
        for i in range(images_num):
            image = load_image("%s/%s" % (args.dataset, train_images[i]))
            n_batch = 1
            count += n_batch

            x_gpu = flow.Tensor(image, requires_grad=True).to("cuda")
            y_origin = transformer(x_gpu)

            x_gpu = utils.normalize_batch(x_gpu)
            y = utils.normalize_batch(y_origin)

            features_x = vgg(x_gpu)
            features_y = vgg(y)
            content_loss = args.content_weight * mse_loss(
                features_y.relu2_2, features_x.relu2_2
            )
            style_loss = 0.0
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            agg_content_loss += content_loss.numpy()
            agg_style_loss += style_loss.numpy()
            if (i + 1) % args.log_interval == 0:
                if args.style_log_dir is not None:
                    y_recover = recover_image(y_origin.numpy())
                    image_recover = recover_image(image)
                    result = np.concatenate(
                        (style_image_recover, image_recover), axis=1
                    )
                    result = np.concatenate((result, y_recover), axis=1)
                    cv2.imwrite(args.style_log_dir + str(i + 1) + ".jpg", result)
                    print(args.style_log_dir + str(i + 1) + ".jpg" + " saved")
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(),
                    e + 1,
                    count,
                    images_num,
                    agg_content_loss[0] / (i + 1),
                    agg_style_loss[0] / (i + 1),
                    (agg_content_loss[0] + agg_style_loss[0]) / (i + 1),
                )
                print(mesg)

            if (
                args.checkpoint_model_dir is not None
                and (i + 1) % args.checkpoint_interval == 0
            ):
                transformer.eval()
                ckpt_model_filename = (
                    "CW_"
                    + str(int(args.content_weight))
                    + "_lr_"
                    + str(args.lr)
                    + "ckpt_epoch"
                    + str(e)
                    + "_"
                    + str(i + 1)
                )
                ckpt_model_path = os.path.join(
                    args.checkpoint_model_dir, ckpt_model_filename
                )
                flow.save(transformer.state_dict(), ckpt_model_path)
                transformer.train()

    # save model
    transformer.eval()
    save_model_filename = (
        "CW_"
        + str(args.content_weight)
        + "_lr_"
        + str(args.lr)
        + "sketch_epoch_"
        + str(args.epochs)
        + "_"
        + str(time.ctime()).replace(" ", "_")
        + "_"
        + str(args.content_weight)
        + "_"
        + str(args.style_weight)
    )
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    flow.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    content_image = load_image_eval(args.content_image)
    with flow.no_grad():
        style_model = TransformerNet()
        state_dict = flow.load(args.model)
        style_model.load_state_dict(state_dict)
        style_model.to("cuda")
        output = style_model(flow.Tensor(content_image).clamp(0, 255).to("cuda"))
    print(args.output_image)
    cv2.imwrite(args.output_image, recover_image(output.numpy()))


def main():
    main_arg_parser = argparse.ArgumentParser(
        description="parser for fast-neural-style"
    )
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser(
        "train", help="parser for training arguments"
    )
    train_arg_parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs, default is 2"
    )
    train_arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for training, default is 4",
    )
    train_arg_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to training dataset, the path should point to a folder "
        "containing another folder with all the training images",
    )
    train_arg_parser.add_argument(
        "--style-image",
        type=str,
        default="images/style-images/mosaic.jpg",
        help="path to style-image",
    )
    train_arg_parser.add_argument(
        "--save-model-dir",
        type=str,
        required=True,
        help="path to folder where trained model will be saved.",
    )
    train_arg_parser.add_argument(
        "--checkpoint-model-dir",
        type=str,
        default=None,
        help="path to folder where checkpoints of trained models will be saved",
    )
    train_arg_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="size of training images, default is 256 X 256",
    )
    train_arg_parser.add_argument(
        "--style-size",
        type=int,
        default=None,
        help="size of style-image, default is the original size of style image",
    )
    train_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU",
    )
    train_arg_parser.add_argument(
        "--seed", type=int, default=42, help="random seed for training"
    )
    train_arg_parser.add_argument(
        "--content-weight",
        type=float,
        default=1e5,
        help="weight for content-loss, default is 1e5",
    )
    train_arg_parser.add_argument(
        "--style-weight",
        type=float,
        default=1e10,
        help="weight for style-loss, default is 1e10",
    )
    train_arg_parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate, default is 1e-3"
    )
    train_arg_parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="number of images after which the training loss is logged, default is 500",
    )
    train_arg_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2000,
        help="number of batches after which a checkpoint of the trained model will be created",
    )
    train_arg_parser.add_argument(
        "--vgg", type=str, default="vgg16", help="choose between vgg16 and vgg19"
    )
    train_arg_parser.add_argument(
        "--style-log-dir",
        type=str,
        default=None,
        help="choose directory to save intermediate style transfer results",
    )
    train_arg_parser.add_argument(
        "--load-checkpoint-dir",
        type=str,
        default=None,
        help="resume training from specified checkpoint directory",
    )

    eval_arg_parser = subparsers.add_parser(
        "eval", help="parser for evaluation/stylizing arguments"
    )
    eval_arg_parser.add_argument(
        "--content-image",
        type=str,
        required=True,
        help="path to content image you want to stylize",
    )
    eval_arg_parser.add_argument(
        "--content-scale",
        type=float,
        default=None,
        help="factor for scaling down the content image",
    )
    eval_arg_parser.add_argument(
        "--output-image",
        type=str,
        required=True,
        help="path for saving the output image",
    )
    eval_arg_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path",
    )
    eval_arg_parser.add_argument(
        "--cuda",
        type=int,
        required=True,
        help="set it to 1 for running on GPU, 0 for CPU",
    )
    eval_arg_parser.add_argument(
        "--export_onnx", type=str, help="export ONNX model to a given file"
    )

    args = main_arg_parser.parse_args()

    
    

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)


if __name__ == "__main__":
    main()
