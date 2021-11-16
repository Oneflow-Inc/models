import os
import oneflow as flow
import oneflow.nn as nn


def make_data_loader(args, mode, is_consistent=False):
    assert mode in ("train", "val")

    if mode == "train":
        total_batch_size = args.batch_size
        batch_size_per_proc = args.batch_size_per_proc
    else:
        total_batch_size = args.val_batch_size
        batch_size_per_proc = args.val_batch_size_per_proc

    placement = None
    sbp = None

    if is_consistent:
        placement = flow.env.all_device_placement("cpu")
        sbp = flow.sbp.split(0)
        batch_size_per_proc = total_batch_size
    
    ofrecord_data_loader = OFRecordDataLoader(
        data_dir=args.data_dir,
        data_part_num=args.data_part_num,
        num_dense_fields=args.num_dense_fields,
        num_wide_sparse_fields=args.num_wide_sparse_fields,
        num_deep_sparse_fields=args.num_deep_sparse_fields,
        batch_size=batch_size_per_proc,
        total_batch_size=total_batch_size,
        mode=mode,
        shuffle=False,
        placement=placement,
        sbp=sbp,
    )
    return ofrecord_data_loader
    


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        data_dir: str = "/dataset/wdl_ofrecord/ofrecord",
        data_part_num: int = 256,
        part_name_suffix_length: int = 5,
        num_dense_fields: int = 13,
        num_wide_sparse_fields: int = 2,
        num_deep_sparse_fields: int = 26,
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

        self.labels = _blob_decoder("labels", (1,))
        self.dense_fields = _blob_decoder(
            "dense_fields", (num_dense_fields,), flow.float
        )
        self.wide_sparse_fields = _blob_decoder(
            "wide_sparse_fields", (num_wide_sparse_fields,)
        )
        self.deep_sparse_fields = _blob_decoder(
            "deep_sparse_fields", (num_deep_sparse_fields,)
        )

    def forward(self):
        reader = self.reader()
        labels = self.labels(reader)
        dense_fields = self.dense_fields(reader)
        wide_sparse_fields = self.wide_sparse_fields(reader)
        deep_sparse_fields = self.deep_sparse_fields(reader)
        return labels, dense_fields, wide_sparse_fields, deep_sparse_fields


if __name__ == "__main__":
    from config import get_args

    FLAGS = get_args()
    dataloader = OFRecordDataLoader(FLAGS, data_root="/dataset/wdl_ofrecord/ofrecord")
    for i in range(10):
        labels, dense_fields, wide_sparse_fields, deep_sparse_fields = dataloader()
        print(deep_sparse_fields)
