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
import imageio
import glob

def is_outlier(points, thresh=3):
    return points >= thresh

def plot(root_dir, epoch):
    g_loss = np.load(os.path.join(root_dir, 'g_loss_{}.npy'.format(epoch)))
    d_loss = np.load(os.path.join(root_dir, 'd_loss_{}.npy'.format(epoch)))
    print("last g_loss: {}.".format(g_loss[-10:-1]))
    print("last d_loss: {}.".format(d_loss[-10:-1]))
    filtered_g_loss = g_loss[~is_outlier(g_loss)]
    filtered_d_loss = d_loss[~is_outlier(d_loss)]
    plt.figure(figsize=(15,5))
    plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(g_loss,label="G")
    # plt.plot(d_loss,label="D")
    plt.plot(filtered_g_loss,label="G")
    plt.plot(filtered_d_loss,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(root_dir, 'loss_{}.png'.format(epoch)))

if __name__ == "__main__":
    root_dir = "/home/zjhuangzhenhua/zjcdy/DCGAN/of_model"
    epoch = 100
    plot(root_dir, epoch)

