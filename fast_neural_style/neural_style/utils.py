import numpy as np
import cv2

import oneflow as flow
import oneflow.typing as tp


def gram_matrix(y):
    (b, ch, h, w) = y.shape
    features = y.reshape((b, ch, w * h))
    features_t = features.transpose(1, 2)
    gram = flow.matmul(features, features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = (
        flow.Tensor([119.90508914, 113.98250597, 103.85173186])
        .reshape((1, 3, 1, 1))
        .to("cuda")
    )
    std = flow.Tensor([58.393, 57.12, 57.375]).reshape((1, 3, 1, 1)).to("cuda")
    return (batch - mean) / std


def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (256, 256))
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def load_image_eval(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def recover_image(im):
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)
