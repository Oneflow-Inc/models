import os
import sys
import glob
import time
import math
import numpy as np
import psutil
import oneflow as flow
from petastorm.reader import make_batch_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class DINDataReader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs=1,
        shuffle_row_groups=False,
        shard_seed=2019,
        shard_count=1,
        cur_shard=0,
        max_len=32,
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_row_groups = shuffle_row_groups
        self.shard_seed = shard_seed
        self.shard_count = shard_count
        self.cur_shard = cur_shard

        fields = ["label", "item_hist", "target", "seq_len"]
        self.fields = fields
        self.num_fields = len(fields)

        self.parquet_file_url_list = parquet_file_url_list
        self.max_len = max_len

    def __enter__(self):
        self.reader = make_batch_reader(
            self.parquet_file_url_list,
            workers_count=1,
            shuffle_row_groups=self.shuffle_row_groups,
            num_epochs=self.num_epochs,
            shard_seed=self.shard_seed,
            shard_count=self.shard_count,
            cur_shard=self.cur_shard,
        )
        self.loader = self.get_batches(self.reader, self.batch_size)
        return self.loader

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()

    def get_batches(self, reader, batch_size=None):
        max_len = self.max_len
        if batch_size is None:
            batch_size = self.batch_size
        tail = None
        for rg in reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in self.fields]
            pos = 0
            if tail is not None:
                pos = batch_size - len(tail[0])
                tail = list(
                    [
                        np.concatenate((tail[i], rglist[i][0 : (batch_size - len(tail[i]))]))
                        for i in range(self.num_fields)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    item_hist = tail[1]
                    target = tail[2]
                    seq_len = tail[3]
                    tail = None
                    yield label, item_hist, target, seq_len
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                item_hist = rglist[1][pos : pos + batch_size]
                target = rglist[2][pos : pos + batch_size]
                seq_len = rglist[3][pos : pos + batch_size]
                pos += batch_size
                yield label, item_hist, target, seq_len
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.num_fields)]
                #tail = [rglist[i][pos:] for i in range(self.C_end)]


def make_dataloader(data_path, batch_size, shuffle=False, max_len=32):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()
    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return DINDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval .dataset
        shuffle_row_groups=shuffle,
        shard_seed=2019,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
        max_len=max_len,
    )


if __name__ == "__main__":
    data_dir = "/data/xiexuan/git-repos/models/RecommenderSystems/din/amazon_elec_parquet"
    with make_dataloader(f"{data_dir}/train", 32, max_len=64) as loader:
        for item_seq, target, label in loader:
            break
            #print(item_seq, target, label)
