import os
import oneflow as flow
import oneflow.nn as nn

class OFRecordDataLoader(nn.Module):
    def __init__(
        self,
        FLAGS,
        batch_size: int = 32,  
        data_root: str = "./ofrecord",
        data_part_num: int = 256,
        part_name_suffix_length: int = 5,
        mode: str = "train",  # "val" "test"
    ):
        super(OFRecordDataLoader, self).__init__()
        assert FLAGS.num_dataloader_thread_per_gpu >= 1
        self.num_dataloader_thread_per_gpu = FLAGS.num_dataloader_thread_per_gpu

        if FLAGS.use_single_dataloader_thread:
            self.devices = ['{}:0'.format(i) for i in range(FLAGS.num_nodes)]
        else:
            num_dataloader_thread = FLAGS.num_dataloader_thread_per_gpu * FLAGS.gpu_num_per_node
            self.devices = ['{}:0-{}'.format(i, num_dataloader_thread - 1) for i in range(FLAGS.num_nodes)]

        self.batch_size = batch_size
        # with flow.scope.placement("cpu", self.devices):
        self.reader = nn.OfrecordReader(
            os.path.join(data_root, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
        )

        def _blob_decoder(bn, shape, dtype=flow.int32):
            return nn.OfrecordRawDecoder(bn, shape=shape, dtype=dtype)
        self.labels = _blob_decoder("labels", (1,))
        self.dense_fields = _blob_decoder("dense_fields", (FLAGS.num_dense_fields,), flow.float)
        self.wide_sparse_fields = _blob_decoder("wide_sparse_fields", (FLAGS.num_wide_sparse_fields,))
        self.deep_sparse_fields = _blob_decoder("deep_sparse_fields", (FLAGS.num_deep_sparse_fields,))

    def forward(self):
        reader = self.reader()
        labels = self.labels(reader)
        dense_fields = self.dense_fields(reader)
        wide_sparse_fields = self.wide_sparse_fields(reader)
        deep_sparse_fields = self.deep_sparse_fields(reader)
        return labels, dense_fields, wide_sparse_fields, deep_sparse_fields
        # return flow.identity_n([labels, dense_fields, wide_sparse_fields, deep_sparse_fields])


if __name__ == '__main__':
    from config import get_args

    FLAGS = get_args()
    
    dataloader = OFRecordDataLoader(FLAGS, ofrecord_root="/dataset/wdl_ofrecord/ofrecord")
    for i in range(10):
        labels, dense_fields, wide_sparse_fields, deep_sparse_fields = dataloader()
        print("iter ", i)
        print("labels.shape =", labels.shape)
        print("dense_fields.shape =", dense_fields.shape)
        print("wide_sparse_fields.shape =", wide_sparse_fields.shape)
        print("deep_sparse_fields.shape =", deep_sparse_fields.shape)
