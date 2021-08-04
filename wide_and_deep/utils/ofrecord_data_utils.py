import oneflow as flow

import os
import glob


class OFRecordDataLoader(object):
    def __init__(
        self,
        FLAGS,
        ofrecord_root: str = "./ofrecord",
        data_part_num: int = 1,
        part_name_suffix_length: int = -1,
        mode: str = "train",  # "val" "test"
        batch_size: int = 1,  
    ):
        assert FLAGS.num_dataloader_thread_per_gpu >= 1
        self.num_dataloader_thread_per_gpu = FLAGS.num_dataloader_thread_per_gpu

        if FLAGS.use_single_dataloader_thread:
            self.devices = ['{}:0'.format(i) for i in range(FLAGS.num_nodes)]
        else:
            num_dataloader_thread = FLAGS.num_dataloader_thread_per_gpu * FLAGS.gpu_num_per_node
            self.devices = ['{}:0-{}'.format(i, num_dataloader_thread - 1) for i in range(FLAGS.num_nodes)]

        self.data_dir = os.path.join(ofrecord_root, mode)
        self.batch_size = batch_size
        self.data_part_num = data_part_num
        self.part_name_suffix_length = part_name_suffix_length
        self.shuffle = mode == "train"

        self.train_record_reader = flow.nn.OfrecordReader(
            os.path.join(ofrecord_root, mode),
            batch_size=batch_size,
            data_part_num=data_part_num,
            part_name_suffix_length=part_name_suffix_length,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
        )

        self.num_dense_fields = FLAGS.num_dense_fields
        self.num_wide_sparse_fields = FLAGS.num_wide_sparse_fields
        self.num_deep_sparse_fields = FLAGS.num_deep_sparse_fields

    def forward(self):
        with flow.scope.placement("cpu", self.devices):
            return flow.identity_n(self.forward_fn())

    def _data_loader_ofrecord(self):
        print('load ofrecord data form', self.data_dir)
        reader = nn.OfrecordReader(self.data_dir,
                                            batch_size=self.batch_size,
                                            data_part_num=self.data_part_num,
                                            part_name_suffix_length=self.part_name_suffix_length,
                                            random_shuffle=self.shuffle,
                                            shuffle_after_epoch=self.shuffle)
        def _blob_decoder(bn, shape, dtype=flow.int32):
            return nn.OfrecordRawDecoder(bn, shape=shape, dtype=dtype)
        labels = _blob_decoder("labels", (1,))
        dense_fields = _blob_decoder("dense_fields", (self.num_dense_fields,), flow.float)
        wide_sparse_fields = _blob_decoder("wide_sparse_fields", (self.num_wide_sparse_fields,))
        deep_sparse_fields = _blob_decoder("deep_sparse_fields", (self.num_deep_sparse_fields,))
        return reader, [labels, dense_fields, wide_sparse_fields, deep_sparse_fields]

