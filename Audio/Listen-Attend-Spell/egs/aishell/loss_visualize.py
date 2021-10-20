#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("Speech-Transformer visualization.")
# General config
parser.add_argument(
    "--tr_loss_path",
    type=str,
    default="exp/temp/tr_loss.npy",
)
parser.add_argument(
    "--val_loss_path",
    type=str,
    default="exp/temp/val_loss.npy",
)
parser.add_argument(
    "--save_figure_name", type=str, default="tr_val_loss.png",
)


def loss_visualize(args):
    """
    Draw loss and accuracy curve
    Args:
        tr_loss_path: the path of training loss values
        val_loss_path: the path of validation loss values
    """
    tr_loss = np.load(args.tr_loss_path, allow_pickle=True)
    tr_loss = list(tr_loss)
    val_loss = np.load(args.val_loss_path, allow_pickle=True)
    val_loss = list(val_loss)
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.grid(1)
    plt.title("Speech Transformer Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.savefig(args.save_figure_name, bbox_inches="tight")


args = parser.parse_args()
print(args)
loss_visualize(args)
