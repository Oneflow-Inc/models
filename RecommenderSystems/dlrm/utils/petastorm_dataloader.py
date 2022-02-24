import os
import oneflow as flow
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader


def make_petastorm_dataloader(args, mode):
    assert mode in ("train", "val")
    return PetastormDataLoader(
        args.data_dir,
        args.train_sub_folders if mode=='train' else args.val_sub_folders,
        num_dense_fields=args.num_dense_fields,
        num_sparse_fields=args.num_sparse_fields,
        batch_size=args.batch_size_per_proc if mode=='train' else args.eval_batch_size_per_proc,
        mode=mode,
    )


class PetastormDataLoader():
    def __init__(
        self,
        data_dir, subfolders,
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 16,
        mode: str = "train",
    ):
        assert mode in ("train", "val")

        self.placement = flow.env.all_device_placement("cuda")
        self.sbp = flow.sbp.split(0)

        files = []
        for folder in subfolders:
            files += ['file://' + name for name in glob.glob(f'{data_dir}/{folder}/*.parquet')]
        files.sort()

        self.reader = make_batch_reader(files, workers_count=2, 
            shuffle_row_groups=(mode=='train'), 
            num_epochs=None if mode == 'train' else 1,
            shard_seed=1234,
            shard_count=flow.env.get_world_size(),
            cur_shard=flow.env.get_rank(),
        )
        self.batch_size = batch_size
        # self.total_batch_size = total_batch_size
        fields = ['label']
        fields += [f"I{i+1}" for i in range(num_dense_fields)]
        self.I_end = len(fields)
        fields += [f"C{i+1}" for i in range(num_sparse_fields)]
        self.C_end = len(fields)
        self.fields = fields
        self.batch_generator = self.get_batches()

    def __call__(self):
        np_label, np_denses, np_sparses = next(self.batch_generator)
        np_dense = np.stack(np_denses, axis=-1)
        np_sparse = np.stack(np_sparses, axis=-1)
        labels = flow.tensor(np_label.reshape(-1, 1), dtype=flow.float)
        dense_fields = flow.tensor(np_dense, dtype=flow.float)
        sparse_fields = flow.tensor(np_sparse, dtype=flow.int32)
        labels = labels.to_global(placement=self.placement, sbp=self.sbp)
        dense_fields = dense_fields.to_global(placement=self.placement, sbp=self.sbp)
        sparse_fields = sparse_fields.to_global(placement=self.placement, sbp=self.sbp)
        return labels, dense_fields, sparse_fields

    def get_batches(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        tail = None
        for rg in self.reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in self.fields]
            pos = 0
            if tail is not None:
                pos = self.batch_size - len(tail[0])
                tail = list([np.concatenate((tail[i], rglist[i][0:(batch_size - len(tail[i]))])) for i in range(self.C_end)])
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    dense = tail[1:self.I_end] #np.stack(tail[1:14], axis=-1)
                    sparse = tail[self.I_end:self.C_end] #np.stack(tail[14:40], axis=-1)
                    tail = None
                    yield label, dense, sparse
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos:pos+batch_size]
                dense = [rglist[j][pos:pos+batch_size] for j in range(1, self.I_end)] 
                sparse = [rglist[j][pos:pos+batch_size] for j in range(self.I_end, self.C_end)] 
                pos += batch_size
                yield label, dense, sparse
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.C_end)]
    
    # def __exit__(self):
    #     self.reader.stop()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_dir = '/minio/sdd/dataset/criteo1t/add_slot_size_snappy_true'
    args.train_sub_folders = [f'day_{i}' for i in range(23)]
    args.val_sub_folders = ['day_23']
    args.num_dense_fields = 13
    args.num_sparse_fields = 26
    args.batch_size_per_proc = 16
    args.eval_batch_size_per_proc = 32

    # subfolders = 
    m = make_petastorm_dataloader(args, mode='train')
    # m = PetastormDataLoader(data_dir, subfolders)
    for i in range(10):
        labels, dense_fields, sparse_fields = m()
        print(i, labels.shape, dense_fields.shape, sparse_fields.shape)
        print(i, labels.is_global)
