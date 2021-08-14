"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def is_outlier(points, thresh=3):
    return points >= thresh


def plot(root_dir, epoch):
    g_loss_graph = np.load(os.path.join(root_dir, "g_loss_graph.npy".format(epoch)))
    d_loss_graph = np.load(os.path.join(root_dir, "d_loss_graph.npy".format(epoch)))
    g_loss = np.load(os.path.join(root_dir, "g_loss.npy".format(epoch)))
    d_loss = np.load(os.path.join(root_dir, "d_loss.npy".format(epoch)))
    print("last g_loss: {}.".format(g_loss[-10:-1]))
    print("last d_loss: {}.".format(d_loss[-10:-1]))
    plt.figure(figsize=(15, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.plot(g_loss_graph, label="G_graph")
    plt.plot(d_loss_graph, label="D_graph")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(root_dir, "loss_parallel.png".format(epoch)))


if __name__ == "__main__":
    root_dir = "./dcgan"
    epoch = 100
    plot(root_dir, epoch)
