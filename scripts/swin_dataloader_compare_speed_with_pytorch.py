import os
import time
import numpy as np
import argparse

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data import create_transform

class SubsetRandomSampler(flow.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in flow.randperm(len(self.indices)).tolist())

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

def build_transform():
    # this should always dispatch to transforms_imagenet_train
    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )
    return transform

# swin-transformer imagenet dataloader
def build_dataset(imagenet_path):
    transform = build_transform()
    prefix = 'train'
    print("build_dataset >>>>>> ImageFolder")
    root = os.path.join(imagenet_path, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_loader(imagenet_path, batch_size, num_wokers):
    dataset_train = build_dataset(imagenet_path=imagenet_path)

    indices = np.arange(flow.env.get_rank(), len(dataset_train), flow.env.get_world_size())
    sampler_train = SubsetRandomSampler(indices)

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_wokers,
        drop_last=True,
    )

    return dataset_train, data_loader_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imagenet_path", type=str)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("num_workers", type=int)
    args = parser.parse_args()

    dataset_train, data_loader_train = build_loader(args.imagenet_path, args.batch_size, args.num_workers)
    data_loader_train_iter = iter(data_loader_train)
    start_time = time.time()
    for idx in range(200):
        samples, targets = data_loader_train_iter.__next__()
    total_time = time.time() - start_time
    print(total_time)
