# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
import os
import random


def load_image(image_path="data/fish.jpg"):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = Image.open(image_path)
    im = im.resize((256, 256))
    im = im.convert("RGB")
    im = np.array(im).astype("float32")
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def image_transform(im):
    rgb_mean = [123.68, 116.779, 103.939]
    rgb_std = [58.393, 57.12, 57.375]
    im = im.resize((256, 256))
    im = im.convert("RGB")
    im = np.array(im).astype("float32")
    im = (im - rgb_mean) / rgb_std
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


class NumpyDataLoader(object):
    def __init__(self, dataset_root: str, batch_size: int = 1):
        self.dataset_root = dataset_root
        sub_folders = os.listdir(self.dataset_root)
        self.image_2_class_label_list = []
        self.label_2_class_name = {}
        self.batch_size = batch_size

        label = -1
        for sf in sub_folders:
            label += 1
            self.label_2_class_name[label] = sf
            sub_root = os.path.join(self.dataset_root, sf)
            image_names = os.listdir(sub_root)
            for name in image_names:
                self.image_2_class_label_list.append(
                    (os.path.join(sub_root, name), label)
                )

        self.curr_idx = 0
        self.shuffle_data()

    def shuffle_data(self):
        random.shuffle(self.image_2_class_label_list)
        self.curr_idx = 0

    def __getitem__(self, index):
        batch_datas = []
        batch_labels = []
        for i in range(self.batch_size):
            image_path, label = self.image_2_class_label_list[self.curr_idx]
            batch_datas.append(load_image(image_path))
            batch_labels.append(int(label))
            self.curr_idx += 1

        np_datas = np.concatenate(tuple(batch_datas), axis=0)
        np_labels = np.array(batch_labels, dtype=np.int32)

        return np.ascontiguousarray(np_datas, "float32"), np_labels

    def __len__(self):
        return len(self.image_2_class_label_list) // self.batch_size


class face_seg(object):
    def __init__(
        self, dataset_root: str, batch_size: int = 1, augmentation=None, training=True
    ):
        self.dataset_root = dataset_root
        sub_folders = os.listdir(self.dataset_root)
        self.image_2_class_label_list = []
        self.label_2_class_name = {}
        self.batch_size = batch_size
        if training:
            self.images = np.array(np.load(self.dataset_root + "img_train.npy"))
            self.labels = np.array(np.load(self.dataset_root + "mask_train.npy"))
        else:
            self.images = np.array(np.load(self.dataset_root + "img_test.npy"))
            self.labels = np.array(np.load(self.dataset_root + "mask_test.npy"))
        self.augmentation = augmentation

        self.curr_idx = 0
        self.shuffle_data()

    def shuffle_data(self):
        random.shuffle(self.image_2_class_label_list)
        self.curr_idx = 0

    def __getitem__(self, index):
        batch_datas = []
        batch_labels = []
        for i in range(self.batch_size):
            _image = self.images[self.curr_idx]
            _label = self.labels[self.curr_idx]
            if self.augmentation is not None:
                data = {"image": _image, "mask": _label}
                augmented = self.augmentation(**data)
                _image, _label = augmented["image"], augmented["mask"]

            batch_datas.append(_image)
            batch_labels.append(_label)
            self.curr_idx += 1
        np_datas = np.array(batch_datas)
        np_labels = np.array(batch_labels, dtype=np.int32)

        return np.ascontiguousarray(np_datas, "float32"), np_labels

    def __len__(self):
        return self.images.shape[0] // self.batch_size
