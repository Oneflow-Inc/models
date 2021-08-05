import oneflow as flow
import oneflow.nn as nn

from config import get_args

class GPTDataLoader(nn.Module):
    def __init__(self, name):
        super(GPTDataLoader, self).__init__()
        self.name = name
        args = get_args()
        self.src = self.SrcModule(args)
        self.data = self.DataModule(args)
        self.label = self.LabelModule(args)

    class SrcModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            assert args.dataset is not None
            self.dataset = args.dataset
            self.batch_size = args.global_batch_size // args.num_accumulation_steps
            self.seq_length = args.seq_length
            self.seed = args.seed
            self.split = args.split
            self.num_samples = args.train_samples
        
        def forward(self):
            x = flow.data.megatron_gpt_mmap_data_loader(
                data_file_prefix=self.dataset,
                seq_length=self.seq_length,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                dtype=flow.int64,
                shuffle=True,
                random_seed=self.seed,
                split_sizes=self.split,
                split_index=0,
                parallel_distribution=distribute.get_data_parallel_dist(),
                name=self.name,
            )
            return x

    class DataModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.seq_length = args.seq_length
        
        def forward(self, x):
            data = flow.slice(x, begin=(None, 0), size=(None, self.seq_length))
            return data

    class LabelModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.seq_length = args.seq_length
        
        def forward(self, x):
            labels = flow.slice(x, begin=(None, 1), size=(None, self.seq_length))
            return labels

    def forward(self):
        # TODO(): 迁移到module
        # TODO(dis, done): 数据特有的placement
        # with distribute.data_placement_scope():
        x = self.src()

        # embedding is on pipeline first stage
        # TODO(dis, done)
        # with distribute.layer_placement_scope(0):
        data = self.data(x)

        # loss is on pipeline last stage
        # TODO(dis, done)
        # with distribute.layer_placement_scope(-1):
        labels = self.label(x)

        return data, labels