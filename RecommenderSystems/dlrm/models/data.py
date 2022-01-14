import os
import oneflow as flow
import oneflow.nn as nn
import glob


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
    if data_format == "ofrecord":
        if mode == "val":
            data_part_num = 256
        else:
            data_part_num = args.data_part_num
        return OFRecordDataLoader(
            data_dir=args.data_dir,
            data_part_num=data_part_num,
            part_name_suffix_length=args.data_part_name_suffix_length,
            **kps
        )
    elif data_format == "onerec":
        return OneRecDataLoader(data_dir=args.data_dir, **kps)
    elif data_format == "synthetic":
        return SyntheticDataLoader(**kps)
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
                    high=150000,
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
                0, high=150000, size=self.sparse_fields_shape, dtype=flow.int32, device="cpu",
            )
            

    def forward(self):
        return self.labels, self.dense_fields, self.sparse_fields

