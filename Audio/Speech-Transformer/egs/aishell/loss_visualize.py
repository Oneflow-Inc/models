#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def loss_visualize(tr_loss_path, val_loss_path):
    """
    Draw loss and accuracy curve
    Args:
        tr_loss_path: the path of training loss values
        val_loss_path: the path of validation loss values
    """
    tr_loss = np.load(tr_loss_path, allow_pickle=True)
    tr_loss = list(tr_loss)
    val_loss = np.load(val_loss_path, allow_pickle=True)
    val_loss = list(val_loss)
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.grid(1)
    plt.title("Speech Transformer Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.savefig("tr_cal_loss.png", bbox_inches="tight")


# loss_visualize('exp/train_m4_n3_in80_elayer6_head8_k64_v64_model512_inner2048_drop0.1_pe5000_emb512_dlayer6_share1_ls0.1_epoch150_shuffle1_bs16_bf15000_mli800_mlo150_k0.2_warm4000/tr_loss.npy', 'egs/aishell/exp/train_m4_n3_in80_elayer6_head8_k64_v64_model512_inner2048_drop0.1_pe5000_emb512_dlayer6_share1_ls0.1_epoch150_shuffle1_bs16_bf15000_mli800_mlo150_k0.2_warm4000/val_loss.npy')
data = np.load(
    "/home/zhaoying/train_m4_n3_in80_elayer6_head8_k64_v64_model512_inner2048_drop0.1_pe5000_emb512_dlayer6_share1_ls0.1_epoch150_shuffle1_bs16_bf15000_mli800_mlo150_k0.2_warm4000/tr_loss.npy"
)
print(len(data))
