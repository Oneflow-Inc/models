import oneflow as flow

from oneflow_gpt.config import get_args
from oneflow_gpt import distribute as dist


class GPTDataLoader(flow.nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args()
        assert args.dataset is not None

        batch_size = args.global_batch_size // args.num_accumulation_steps
        self.reader_ = flow.nn.GPTIndexedBinDataReader(
            data_file_prefix=args.dataset,
            seq_length=args.seq_length,
            num_samples=args.train_samples,
            batch_size=batch_size,
            dtype=flow.int64,
            shuffle=True,
            random_seed=args.seed,
            split_sizes=args.split,
            split_index=0,
            placement=dist.get_layer_placement(-1, "cpu"),
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
        )

    def forward(self):
        tokens = self.reader_()
        tokens = tokens.to("cuda")
        data = tokens[:, 0:-1]
        # loss is on pipeline last stage
        labels = tokens[:, 1:].to_consistent(placement=dist.get_layer_placement(-1))
        return data, labels
