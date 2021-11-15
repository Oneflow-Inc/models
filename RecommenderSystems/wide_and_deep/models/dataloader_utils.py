import os
import oneflow as flow
import oneflow.nn as nn


class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        FLAGS,
        data_part_num: int = 256,
        part_name_suffix_length: int = 5,
        mode: str = "train",
    ):
        super(OFRecordDataLoader, self).__init__()
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        print(self.rank, self.world_size)
        print('all_device_placement', flow.env.all_device_placement('cpu'))
        print('all_device_placement', flow.env.all_device_placement('cuda'))
        print('get_local_rank', flow.env.get_local_rank())
        print('get_node_size', flow.env.get_node_size())
        print('machine', flow.env.machine())
        print('*'*100)

        data_root = FLAGS.data_dir
        batch_size = FLAGS.batch_size
        is_consistent = (
            flow.env.get_world_size() > 1 and not FLAGS.ddp
        ) or FLAGS.execution_mode == "graph"
        placement = None
        sbp = None
        if is_consistent == True:
            placement = flow.placement("cpu", {0: range(flow.env.get_world_size())})
            sbp = flow.sbp.split(0)
        shuffle = mode == "train"
        self.reader = nn.OfrecordReader(
            os.path.join(data_root, mode),
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
            "dense_fields", (FLAGS.num_dense_fields,), flow.float
        )
        self.wide_sparse_fields = _blob_decoder(
            "wide_sparse_fields", (FLAGS.num_wide_sparse_fields,)
        )
        self.deep_sparse_fields = _blob_decoder(
            "deep_sparse_fields", (FLAGS.num_deep_sparse_fields,)
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
    dataloader = OFRecordDataLoader(FLAGS)
    # for i in range(10):
    #     labels, dense_fields, wide_sparse_fields, deep_sparse_fields = dataloader()
    #     print(deep_sparse_fields)
