import os, sys
import numpy as np
import oneflow.experimental as flow
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from io import TextIOBase
import time

def to_tensor(x, grad=False, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.Tensor(x, requires_grad=grad, dtype=dtype).to("cuda")

def to_numpy(x, mean=True):
    if mean:
        x = flow.mean(x)
    return x.numpy()

def save_images(images, real_input, target, path, plot_size=12):
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

    plt.savefig(path)
    plt.close()

log_level_map = {
    "fatal": logging.FATAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

_time_format = "%m/%d/%Y, %I:%M:%S %p"

class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != "\n":
            cur_time = datetime.now().strftime(_time_format)
            self.file.write("[{}] PRINT ".format(cur_time) + s + "\n")
            self.file.flush()
        return len(s)


def init_logger(logger_file_path, log_level_name="info"):

    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    log_level = log_level_map.get(log_level_name)
    logger_file = open(logger_file_path, "w")
    fmt = "[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s"

    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(logger_file_path)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(log_level)

    # include print function output
    sys.stdout = _LoggerFileWrapper(logger_file)

    return root_logger


def mkdirs(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)
