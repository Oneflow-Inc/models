import itertools
import os

import numpy as np
import cv2


def augment(img, random_scale_limit=0.1, resize_smallest_side=270, random_crop_h_w=256, rng=None):
    assert resize_smallest_side > random_crop_h_w

    # img: float32 HxWxC
    assert img.shape[2] == 3

    if rng is None:
        rng = np.random.default_rng()

    # normalize: [0, 1] -> [-1, 1]
    img = (img - np.float32(0.5)) * np.float32(2)

    # resize_smallest_side
    h, w, _ = img.shape
    scale = max(resize_smallest_side / h, resize_smallest_side / w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)

    # random_scale_limit
    img *= np.float32(rng.uniform(1, 1+random_scale_limit))

    # horizontal_flip
    if rng.integers(0, 2):
        img = np.flip(img, axis=1)
    
    # random_crop_h_w
    crop_y = rng.integers(0, img.shape[0]-random_crop_h_w)
    crop_x = rng.integers(0, img.shape[1]-random_crop_h_w)
    img = img[crop_y:crop_y+random_crop_h_w, crop_x:crop_x+random_crop_h_w, :]

    return img


class Dataset:
    def __init__(self, path, augment=augment, seed=None):
        assert os.path.isdir(path)

        raw_class_labels = os.listdir(path)

        self.path = path

        self.class_idx_mapping = {
            class_name : idx 
            for idx, class_name in enumerate(raw_class_labels)
        }

        self.img_paths = list(
            itertools.chain.from_iterable(
                os.listdir(os.path.join(path, class_label)) 
                for class_label in raw_class_labels
            )
        )

        self.augment = augment

        self.rng = np.random.default_rng(seed)

        self.shuffle()

    def __getitem__(self, index):
        path = self.img_paths[index]

        raw_label = path[:path.index('_')]
        label = self.class_idx_mapping[raw_label]

        img = cv2.imread(
            os.path.join(self.path, raw_label, path), 
            cv2.IMREAD_COLOR
        )
        img = img.astype(np.float32) / np.float32(255)
        img = self.augment(img, rng=self.rng)
        img = np.transpose(img, (2, 0, 1))

        return img, label

    def __len__(self):
        return len(self.img_paths)

    def shuffle(self):
        self.rng.shuffle(self.img_paths)

    def data_iterator(self, batch_size):
        for i in range(0, len(self) - batch_size + 1, batch_size):
            data = [self[j] for j in range(i, i + batch_size)]

            raw_imgs = [img for img, _ in data]
            imgs = [np.expand_dims(img, axis=0) for img in raw_imgs]
            if batch_size > 1:
                img_batch = np.concatenate(imgs, axis=0)
            else:
                img_batch = imgs[0]

            raw_labels = [np.int32(label) for _, label in data]
            labels = [np.expand_dims(label, axis=0) for label in raw_labels]
            if batch_size > 1:
                label_batch = np.concatenate(labels, axis=0)
            else:
                label_batch = labels[0]

            yield img_batch, label_batch
