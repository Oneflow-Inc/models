import os
import time
import numpy as np
import argparse

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data import create_transform


ONEREC_URL = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/nanodataset.zip"
MD5 = "7f5cde8b5a6c411107517ac9b00f29db"


def md5(fname):
    import hashlib

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    result = hash_md5.hexdigest()
    return result


def download_file(out_path: str, url):
    import requests
    from tqdm import tqdm

    resp = requests.get(url=url, stream=True)
    MB = 1024 ** 2
    size = int(resp.headers["Content-Length"]) / MB
    print("File size: %.4f MB, downloading..." % size)
    with open(out_path, "wb") as f:
        for data in tqdm(
            iterable=resp.iter_content(MB), total=size, unit="m", desc=out_path
        ):
            f.write(data)
        print("Done!")


def ensure_dataset():
    import os
    import pathlib

    data_dir = os.path.join(
        os.getenv("ONEFLOW_TEST_CACHE_DIR", "./data-test"), "onerec_test"
    )
    file_path = pathlib.Path(data_dir) / ONEREC_URL.split("/")[-1]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    absolute_file_path = str(file_path.absolute())
    if file_path.exists():
        if MD5 != md5(absolute_file_path):
            file_path.unlink()
            download_file(absolute_file_path, ONEREC_URL)
    else:
        download_file(str(absolute_file_path), ONEREC_URL)
    assert MD5 == md5(absolute_file_path)
    import shutil
    import pathlib

    shutil.unpack_archive(absolute_file_path)
    return str(pathlib.Path.cwd() / "nanodataset")


def print_rank_0(*args, **kwargs):
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


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
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
    )
    return transform


# swin-transformer imagenet dataloader
def build_dataset(imagenet_path):
    transform = build_transform()
    prefix = "train"
    root = os.path.join(imagenet_path, prefix)
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_loader(imagenet_path, batch_size, num_wokers):
    dataset_train = build_dataset(imagenet_path=imagenet_path)

    indices = np.arange(
        flow.env.get_rank(), len(dataset_train), flow.env.get_world_size()
    )
    sampler_train = SubsetRandomSampler(indices)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=batch_size,
        num_workers=num_wokers,
        drop_last=True,
    )

    return dataset_train, data_loader_train


def run(mode, imagenet_path, batch_size, num_wokers):
    if mode == "torch":
        import torch as flow
        from torch.utils.data import DataLoader

        from torchvision import datasets, transforms
        from timm.data import create_transform

    dataset_train, data_loader_train = build_loader(
        args.imagenet_path, args.batch_size, args.num_workers
    )
    data_loader_train_iter = iter(data_loader_train)

    # warm up
    for idx in range(5):
        samples, targets = data_loader_train_iter.__next__()

    start_time = time.time()
    for idx in range(200):
        samples, targets = data_loader_train_iter.__next__()
    total_time = time.time() - start_time
    return total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--imagenet_path", type=str, required=False)
    args = parser.parse_args()
    if not args.imagenet_path:
        args.imagenet_path = ensure_dataset()
    oneflow_data_loader_time = run(
        "oneflow", args.imagenet_path, args.batch_size, args.num_workers
    )
    pytorch_data_loader_time = run(
        "torch", args.imagenet_path, args.batch_size, args.num_workers
    )
    relative_speed = oneflow_data_loader_time / pytorch_data_loader_time

    print_rank_0(
        f"Swin Transformer dataloader relative speed when num_workers = {args.num_workers}: {relative_speed:.2f} (= {pytorch_data_loader_time:.1f}s / {oneflow_data_loader_time:.1f}s)"
    )
