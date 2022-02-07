import os
import oneflow as flow
import oneflow.nn as nn
import glob
from petastorm.reader import make_batch_reader
import numpy as np
import time

__all__ = ["make_data_loader"]

def make_data_loader(args, mode, is_consistent=False, data_format="ofrecord"):
    assert mode in ("train", "val")

    total_batch_size = args.batch_size
    batch_size_per_proc = args.batch_size_per_proc
    eval_total_batch_size = args.eval_batch_size
    eval_batch_size_per_proc = args.eval_batch_size_per_proc

    placement = None
    sbp = None

    if is_consistent:
        placement = flow.env.all_device_placement("cpu")
        sbp = flow.sbp.split(0)
        batch_size_per_proc = total_batch_size
        eval_batch_size_per_proc = eval_total_batch_size
    
    kps = dict(
        num_dense_fields=args.num_dense_fields,
        num_sparse_fields=args.num_sparse_fields,
        batch_size=batch_size_per_proc if mode=='train' else eval_batch_size_per_proc,
        total_batch_size=total_batch_size if mode=='train' else eval_total_batch_size,
        mode=mode,
        shuffle=(mode=='train'),   
        placement=placement,
        sbp=sbp,
    )
    if data_format == "parquet":
        return ParquetDataLoader(data_dir=args.data_dir, **kps)
    elif data_format == "ofrecord":
        return OFRecordDataLoader(
            data_dir=args.data_dir,
            data_part_num=args.data_part_num if mode=='train' else args.eval_data_part_num,
            part_name_suffix_length=args.data_part_name_suffix_length,
            **kps
        )
    elif data_format == "onerec":
        return OneRecDataLoader(data_dir=args.data_dir, **kps)
    elif data_format == "synthetic":
        return SyntheticDataLoader(**kps)
    elif data_format == "petastorm":
        return PetastormDataLoader(**kps)
    else:
        raise ValueError("data format must be one of ofrecord, onerec or synthetic")


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_ofrecord/ofrecord",
        data_part_num: int = 256,
        part_name_suffix_length: int = 5,
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,
        placement=None,
        sbp=None,
    ):
        super(OFRecordDataLoader, self).__init__()
        assert mode in ("train", "val")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.mode = mode
  
        self.reader = nn.OfrecordReader(
            os.path.join(data_dir, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=shuffle,
            shuffle_after_epoch=shuffle,
            placement=placement,
            sbp=sbp,
        )

        def _blob_decoder(bn, shape, dtype=flow.int32):
            return nn.OfrecordRawDecoder(bn, shape=shape, dtype=dtype)

        self.labels = _blob_decoder("labels", (1,), flow.float)
        self.dense_fields = _blob_decoder(
            "dense_fields", (num_dense_fields,), flow.float
        )
        self.sparse_fields = _blob_decoder(
            "deep_sparse_fields", (num_sparse_fields,)
        )

    def forward(self):
        reader = self.reader()
        labels = self.labels(reader)
        dense_fields = self.dense_fields(reader)
        sparse_fields = self.sparse_fields(reader)
        print("dense_fields shape", dense_fields.shape)
        return labels, dense_fields, sparse_fields


class OneRecDataLoader(nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_onerec",
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,
        placement=None,
        sbp=None,
    ):
        super(OneRecDataLoader, self).__init__()
        assert mode in ("train", "val")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.num_dense_fields = num_dense_fields
        self.num_sparse_fields = num_sparse_fields
        self.mode = mode
        self.placement = placement
        self.sbp = sbp
        self.shuffle = shuffle
        self.onerec_files = glob.glob(os.path.join(data_dir, mode, '*.onerec'))

    def _blob_decoder(self, reader, bn, shape, dtype=flow.int32):
        return flow.decode_onerec(reader, bn, shape=shape, dtype=dtype)

    def forward(self):
        reader = flow.read_onerec(
            self.onerec_files,
            batch_size=self.batch_size,
            random_shuffle=self.shuffle,
            verify_example=False,
            shuffle_mode="batch",
            shuffle_buffer_size=64,
            shuffle_after_epoch=self.shuffle,
            placement = self.placement,
            sbp = self.sbp,
        )
        labels = self._blob_decoder(reader, "labels", (1,), flow.float)
        dense_fields = self._blob_decoder(reader, "dense_fields", (self.num_dense_fields,), flow.float)
        sparse_fields = self._blob_decoder(reader, "deep_sparse_fields", (self.num_sparse_fields,))
        return labels, dense_fields, sparse_fields


class SyntheticDataLoader(nn.Module):
    def __init__(
        self,
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,    
        placement=None,
        sbp=None,
    ):
        super(SyntheticDataLoader, self).__init__()
        print("use synthetic data")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.placement = placement
        self.sbp = sbp

        self.label_shape = (batch_size, 1)
        self.dense_fields_shape = (batch_size, num_dense_fields)
        self.sparse_fields_shape = (batch_size, num_sparse_fields)

        if self.placement is not None and self.sbp is not None:
            self.labels = flow.randint(
                    0,
                    high=2,
                    size=self.label_shape,
                    dtype=flow.float,
                    placement=self.placement,
                    sbp=self.sbp,
            )

            self.dense_fields = flow.randint(
                    0,
                    high=256,
                    size=self.dense_fields_shape,
                    dtype=flow.float,
                    placement=self.placement,
                    sbp=self.sbp,
            )

            self.sparse_fields = flow.randint(
                    0,
                    high=256,
                    size=self.sparse_fields_shape,
                    dtype=flow.int32,
                    placement=self.placement,
                    sbp=self.sbp,
            )
        else:
            self.labels = flow.randint(
                0, high=2, size=self.label_shape, dtype=flow.float, device="cpu"
            )
            self.dense_fields = flow.randint(
                0, high=256, size=self.dense_fields_shape, dtype=flow.float, device="cpu",
            )
            self.sparse_fields = flow.randint(
                0, high=256, size=self.sparse_fields_shape, dtype=flow.int32, device="cpu",
            )
            

    def forward(self):
        return self.labels, self.dense_fields, self.sparse_fields

class ParquetDataLoader(nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_parquet",
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,
        placement=None,
        sbp=None,
    ):
        super(ParquetDataLoader, self).__init__()
        assert mode in ("train", "val")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        self.mode = mode
        schema = [
            {"col_name": "labels", "shape": (1,), "dtype": flow.double},
            {"col_name": 'dense_fields', "shape": (num_dense_fields,), "dtype": flow.double},
            {"col_name": 'deep_sparse_fields', "shape": (num_sparse_fields,), "dtype": flow.int32},
            # {"col_id": 1, "shape": (num_dense_fields,), "dtype": flow.double},
            # {"col_id": 2, "shape": (num_sparse_fields,), "dtype": flow.int32},
            # {"col_id": 3, "shape": (2,), "dtype": flow.int32},
        ]
        self.reader = nn.ParquetReader(
            os.path.join(data_dir, mode),
            schema=schema,
            batch_size=batch_size,
            shuffle=shuffle,
            placement=placement,
            sbp=sbp,
        )

    def forward(self):
        labels, dense_fields, sparse_fields = self.reader()
        labels = flow.cast(labels, flow.float)
        dense_fields = flow.cast(dense_fields, flow.float)        
        return labels, dense_fields, sparse_fields

files = ['file://' + name for name in glob.glob('/NVME3/liujuncheng/from_ofrecord/day_0/*.parquet')]

fields = ['label']
fields += ["I{}".format(i + 1) for i in range(13)]
fields += ["C{}".format(i + 1) for i in range(26)]
def get_batches(reader, batch_size):
    tail = None
    for rg in reader:
        rgdict = rg._asdict()
        rglist = [rgdict[field] for field in fields]
        pos = 0
        if tail is not None:
            pos = batch_size - len(tail[0])
            tail = list([np.concatenate((tail[i], rglist[i][0:(batch_size - len(tail[i]))])) for i in range(40)])
            if len(tail[0]) == batch_size:
                label = tail[0]
                dense = tail[1:14] #np.stack(tail[1:14], axis=-1)
                sparse = tail[14:40] #np.stack(tail[14:40], axis=-1)
                tail = None
                yield label, dense, sparse
            else:
                pos = 0
                continue
        while (pos + batch_size) <= len(rglist[0]):
            label = rglist[0][pos:pos+batch_size]
            dense = [rglist[j][pos:pos+batch_size] for j in range(1, 14)] #np.stack([rglist[j][pos:pos+batch_size] for j in range(1, 14)], axis=-1)
            sparse = [rglist[j][pos:pos+batch_size] for j in range(14, 40)] #np.stack([rglist[j][pos:pos+batch_size] for j in range(14, 40)], axis=-1)
            pos += batch_size
            yield label, dense, sparse
        if pos != len(rglist[0]):
            tail = [rglist[i][pos:] for i in range(40)]

class PetastormDataLoader(nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_parquet",
        num_dense_fields: int = 13,
        num_sparse_fields: int = 26,
        batch_size: int = 1,
        total_batch_size: int = 1,
        mode: str = "train",
        shuffle: bool = True,
        placement=None,
        sbp=None,
    ):
        super(PetastormDataLoader, self).__init__()
        assert mode in ("train", "val")
        self.batch_size = batch_size
        self.total_batch_size = total_batch_size
        print("batch_size total_batch_size", batch_size, total_batch_size)
        self.mode = mode
        self.reader = make_batch_reader(files, workers_count=2, shuffle_row_groups=False)
        self.placement = placement
        self.sbp = sbp
        #self.placement = flow.env.all_device_placement("cuda")
        #self.sbp = flow.sbp.split(0)
        self.batch_generator = get_batches(self.reader, self.batch_size)

    def forward(self):
        time.sleep(50.0/1000)
        np_label, np_denses, np_sparses  = next(self.batch_generator)
        print("label", np_label.shape)
        print("np_dense", np_denses[0].shape)
        print("np_sparse", np_sparses[0].shape)
        labels = flow.tensor(
                    np_label.reshape(-1,1).astype(np.int32),
                    dtype = flow.int64,
                    placement=self.placement,
                    sbp=self.sbp,
        )
        dense_fields_list = []
        for np_dense in np_denses:
            dense_fields_list.append(flow.tensor(
                        np_dense.reshape(-1,1).astype(np.float32),
                        placement=self.placement,
                        sbp=self.sbp,
            ))
        dense_fields = flow.cat(dense_fields_list, dim=1)
        sparse_fields_list = []
        for np_sparse in np_sparses:
            sparse_fields_list.append(flow.tensor(
                        np_sparse.reshape(-1,1).astype(np.int64),
                        placement=self.placement,
                        sbp=self.sbp,
            ))
        sparse_fields = flow.cat(sparse_fields_list, dim=1)
        return labels, dense_fields, sparse_fields


if __name__ == "__main__":
    m = ParquetDataLoader("/tank/dataset/criteo_kaggle/dlrm_parquet", batch_size=32, total_batch_size=32)
    labels, dense_fields, sparse_fields = m()
    print(labels.shape)
