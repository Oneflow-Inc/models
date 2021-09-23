import os
import time
import argparse
import numpy as np
import glob
import imageio
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import oneflow as flow


def make_dirs(*pathes):
    for path in pathes:
        # dir path
        if not os.path.exists(path):
            os.makedirs(path)


def load_mnist(data_dir, transpose=True):
    if os.path.exists(data_dir):
        print("Found MNIST - skip download")
    else:
        print("not Found MNIST - start download")
        download_mnist(data_dir)

    fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float32)

    X = trX
    y = trY.astype(int)

    seed = 547
    # np.random.seed(seed)
    np.random.shuffle(X)
    # np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    if transpose:
        X = np.transpose(X, (0, 3, 1, 2))

    return (X - 127.5) / 127.5, y_vec


def download_mnist(data_dir):
    import subprocess

    os.mkdir(data_dir)
    url_base = "http://yann.lecun.com/exdb/mnist/"
    file_names = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ["curl", url, "-o", out_path]
        print("Downloading ", file_name)
        subprocess.call(cmd)
        cmd = ["gzip", "-d", out_path]
        print("Decompressing ", file_name)
        subprocess.call(cmd)


def to_numpy(x, mean=True):
    if mean:
        x = flow.mean(x)

    return x.numpy()


def to_tensor(x, grad=True, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.tensor(x, requires_grad=grad, dtype=dtype)


def save_to_gif(path):
    anim_file = os.path.join(path, "dcgan.gif")
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = glob.glob(os.path.join(path, "*image*.png"))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    print("Save images gif to {} done.".format(anim_file))


def save_images(x, size, path):
    x = x.astype(np.float)
    fig = plt.figure(figsize=(4, 4))
    for i in range(size):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i, 0, :, :] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig(path)
    print("Save image to {} done.".format(path))
