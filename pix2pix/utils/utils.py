import os
import numpy as np
import oneflow.experimental as flow
import matplotlib.pyplot as plt


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def to_tensor(x, grad=True, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.Tensor(x, requires_grad=grad, dtype=dtype)

def to_numpy(x, mean=True):
    if mean:
        x = flow.mean(x)
    return x.numpy()
    
def save_images(self, images, real_input, target, epoch_idx, name, path=None):
    if name == "eval":
        plot_size = epoch_idx
    else:
        plot_size = self.batch_size

    if name == "train":
        images_path = self.train_images_path 
    elif name == "test":
        images_path = self.test_images_path

    plt.figure(figsize=(6, 8))
    display_list = list(zip(real_input, target, images))
    # title = ["Input Image", "Ground Truth", "Predicted Image"]
    idx = 1
    row = 4
    # save 4 images of title
    for i in range(plot_size):
        dis = display_list[i]
        for j in range(3):
            plt.subplot(row, 6, idx)
            # plt.title(title[j])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(np.array(dis[j]).transpose(1, 2, 0) * 0.5 + 0.5)
            plt.axis("off")
            idx = idx + 1

        if idx > row * 6:
            break
    if name == "eval":
        save_path = path
    else:
        save_path = os.path.join(images_path, "{}_image_{:02d}.png".format(name, epoch_idx + 1))
    plt.savefig(save_path)
    plt.close()