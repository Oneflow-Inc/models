import oneflow as flow
from oneflow import nn


class OfRecordDataLoader(nn.Module):
    def __init__(
        self,
        ofrecord_dir: str,
        mode: str,
        dataset_size: int,
        batch_size: int,
        data_part_num: int,
        seq_length: int,
        max_predictions_per_seq: int,
        consistent: bool = False,
    ):
        super().__init__()

        self.placement = None
        self.sbp = None
        self.use_consistent = consistent

        if self.use_consistent:
            world_size = flow.env.get_world_size()
            # self.placement = flow.placement("cuda", {0: range(world_size)})
            self.placement = flow.placement("cpu", {0: range(world_size)})
            self.sbp = flow.sbp.split(0)
            # NOTE(zwx): consistent view, only consider logical batch size
            # batch_size = total_batch_size

        self.ofrecord_reader = nn.OfrecordReader(
            ofrecord_dir,
            batch_size=batch_size,
            data_part_num=data_part_num,
            random_shuffle=True if mode == "train" else False,
            shuffle_after_epoch=True if mode == "train" else False,
            placement=self.placement,
            sbp=self.sbp,
        )

        blob_confs = {}

        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = nn.OfrecordRawDecoder(name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("next_sentence_labels", [1])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("masked_lm_ids", [max_predictions_per_seq])
        _blob_conf("masked_lm_positions", [max_predictions_per_seq])
        _blob_conf("masked_lm_weights", [max_predictions_per_seq], flow.float)

        self.blob_confs = blob_confs
        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size // self.batch_size

    def forward(self):
        data_record = self.ofrecord_reader()  # get an item
        input_ids = self.blob_confs["input_ids"](data_record)
        next_sent_labels = self.blob_confs["next_sentence_labels"](data_record)
        input_mask = self.blob_confs["input_mask"](data_record)
        segment_ids = self.blob_confs["segment_ids"](data_record)
        masked_lm_ids = self.blob_confs["masked_lm_ids"](data_record)
        masked_lm_positions = self.blob_confs["masked_lm_positions"](data_record)
        masked_lm_weights = self.blob_confs["masked_lm_weights"](data_record)

        return (
            input_ids,
            next_sent_labels,
            input_mask,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
            masked_lm_weights,
        )
