import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader


class ParquetDataloader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs,
        num_dense_fields=13,
        num_sparse_fields=26,
        shuffle_row_groups=True,
        shard_seed=1234,
        shard_count=1,
        cur_shard=0,
    ):
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_row_groups = shuffle_row_groups
        self.shard_seed = shard_seed
        self.shard_count = shard_count
        self.cur_shard = cur_shard

        fields = ["label"]
        fields += [f"I{i+1}" for i in range(num_dense_fields)]
        self.I_end = len(fields)
        fields += [f"C{i+1}" for i in range(num_sparse_fields)]
        self.C_end = len(fields)
        self.fields = fields

    def __enter__(self):
        self.reader = make_batch_reader(
            self.parquet_file_url_list,
            workers_count=2,
            shuffle_row_groups=self.shuffle_row_groups,
            num_epochs=self.num_epochs,
            shard_seed=self.shard_seed,
            shard_count=self.shard_count,
            cur_shard=self.cur_shard,
        )
        self.loader = self.get_batches(self.reader)
        return self.loader

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()

    def get_batches(self, reader, batch_size=None):
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
                        np.concatenate(
                            (tail[i], rglist[i][0 : (batch_size - len(tail[i]))])
                        )
                        for i in range(self.C_end)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    dense = tail[1 : self.I_end]
                    sparse = tail[self.I_end : self.C_end]
                    tail = None
                    yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                dense = [
                    rglist[j][pos : pos + batch_size] for j in range(1, self.I_end)
                ]
                sparse = [
                    rglist[j][pos : pos + batch_size]
                    for j in range(self.I_end, self.C_end)
                ]
                pos += batch_size
                yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.C_end)]


if __name__ == "__main__":
    import glob

    np.set_printoptions(linewidth=100)
    mode = "train"
    data_dir = "/minio/sdd/dataset/criteo1t/add_slot_size_snappy_true"
    train_sub_folders = [f"day_{i}" for i in range(23)]
    val_sub_folders = ["day_23"]
    subfolders = train_sub_folders if mode == "train" else val_sub_folders

    files = []
    for folder in subfolders:
        files += [
            "file://" + name for name in glob.glob(f"{data_dir}/{folder}/*.parquet")
        ]
    files.sort()
    batch_size = 32
    num_epochs = 1
    with ParquetDataloader(files, batch_size, num_epochs) as dataloader:
        for i in range(10):
            labels, dense_fields, sparse_fields = next(dataloader)
            print(i, labels.shape, dense_fields.shape, sparse_fields.shape)
            print(i, type(labels), type(dense_fields), type(sparse_fields))

    with ParquetDataloader(files, batch_size, num_epochs) as dataloader:
        i = 0
        for labels, dense_fields, sparse_fields in dataloader:
            i += 1
            print(i, labels.shape, dense_fields.shape, sparse_fields.shape)
            print(i, type(labels), type(dense_fields), type(sparse_fields))
            break

    import time
    import psutil

    for i in range(1000):
        with ParquetDataloader(files, batch_size, num_epochs) as dataloader:
            pass
        # time.sleep(0.5)
        print(i, time.time(), psutil.Process().memory_info().rss // (1024 * 1024))
