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


def plot_loss(root_dir, epoch):
    gen_image_loss = np.load(os.path.join(root_dir, 'G_image_loss_{}.npy'.format(epoch)))
    gen_gan_loss = np.load(os.path.join(root_dir, 'G_GAN_loss_{}.npy'.format(epoch)))
    gen_total_loss = np.load(os.path.join(root_dir, 'G_total_loss_{}.npy'.format(epoch)))
    d_loss = np.load(os.path.join(root_dir, 'D_loss_{}.npy'.format(epoch)))
    print("gen_image_loss", gen_image_loss[-10:-1])
    print("gen_total_loss", gen_total_loss[-10:-1])
    print("gen_gan_loss", gen_gan_loss[-10:-1])
    print("d_loss", d_loss[-10:-1])
    loss = [[gen_image_loss, gen_gan_loss, d_loss], gen_total_loss]
    name = [["G_image", "G", "D"], "G_total"]
    # save_path = os.path.join(root_dir, 'loss_{}.png'.format(epoch))
    for lo, na in zip(loss, name):
        plt.figure(figsize=(15,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        if isinstance(lo, list):
            for i in range(len(lo)):
                plt.plot(lo[i],label=na[i])
            plt.legend()
            plt.savefig(os.path.join(root_dir, 'image_loss_{}.png'.format(epoch)))
        else:
            plt.plot(lo,label=na)
            plt.legend()
            plt.savefig(os.path.join(root_dir, '{}_loss_{}.png'.format(na, epoch)))

if __name__ == "__main__":
    root_dir = "../of_pix2pix"
    epoch = 200
    plot_loss(root_dir, epoch)