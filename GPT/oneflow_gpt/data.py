import oneflow as flow

from oneflow_gpt.config import get_args
from oneflow_gpt import distribute as dist


class GPTDataLoader(flow.nn.Module):
    def __init__(self):
        super().__init__()
        args = get_args()
        assert args.dataset is not None

        batch_size = args.global_batch_size // args.num_accumulation_steps
        self.reader = flow.nn.GPTIndexedBinDataReader(
            data_file_prefix=args.dataset,
            seq_length=args.seq_length,
            num_samples=args.train_samples,
            batch_size=batch_size,
            dtype=flow.int64,
            shuffle=True,
            random_seed=args.seed,
            split_sizes=args.split,
            split_index=0,
            placement=dist.get_layer_placement(0, "cpu"),
            sbp=dist.get_nd_sbp([flow.sbp.split(0), flow.sbp.broadcast]),
        )
        self.data_decoder = DataDecoder()
        self.label_decoder = LabelDecoder()

    def forward(self):
        tokens = self.reader()
        data = self.data_decoder(tokens)
        labels = self.label_decoder(tokens)
        return data, labels


class DataDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(0))[:, 0:-1]


class LabelDecoder(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens):
        assert tokens.ndim == 2
        return tokens.to_consistent(placement=dist.get_layer_placement(-1))[:, 1:]
