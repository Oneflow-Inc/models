import os
import sys
import glob
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import psutil

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from config import get_args
from parquet_dataloader import ParquetDataloader
from dlrm import make_dlrm_module
# from lr_scheduler import make_lr_scheduler


def make_criteo_dataloader(args, mode):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader
                will be closed.
    """
    subfolders = args.train_sub_folders if mode=='train' else args.val_sub_folders

    files = []
    for folder in subfolders:
        files += ['file://' + name for name in glob.glob(f'{args.data_dir}/{folder}/*.parquet')]
    files.sort()

    return ParquetDataloader(
        files,
        args.batch_size_per_proc if mode=='train' else args.eval_batch_size_per_proc,
        None, # if mode=='train' else 1, # TODO: iterate over all eval dataset
        num_dense_fields=args.num_dense_fields,
        num_sparse_fields=args.num_sparse_fields,
        shuffle_row_groups=(mode=='train'),
        shard_seed=1234,
        shard_count=flow.env.get_world_size(),
        cur_shard=flow.env.get_rank())


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=args.warmup_batches,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer,
        steps=args.decay_batches,
        end_learning_rate=0,
        power=2.0,
        cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[args.decay_start],
        interval_rescaling=True,
    )
    return sequential_lr


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.save_path = args.model_save_dir
        self.eval_interval = args.eval_interval
        self.eval_batchs = args.eval_batchs

        self.cur_iter = 0
        self.rank = flow.env.get_rank()
        self.placement = flow.env.all_device_placement("cuda")
        self.sbp = flow.sbp.split(0)

        assert args.model_load_dir != "", "model_load_dir can not empty"
        print(f"Loading model from {args.model_load_dir}")
        t0 = time.time()
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        t1 = time.time()
        
        if 'bottom_mlp.linear_layers.weight_0' in state_dict.keys():
            args.mlp_type = "FusedMLP"
        elif 'bottom_mlp.linear_layers.fc0.features.0.weight' in state_dict.keys():
            args.mlp_type = "MLP"
        else:
            assert "unsupported state dict"

        self.dlrm_module = make_dlrm_module(args)
        self.dlrm_module.to_global(self.placement, flow.sbp.broadcast)
        self.dlrm_module.embedding.set_model_parallel(self.placement)

        self.dlrm_module.load_state_dict(state_dict, strict=True)
        t2 = time.time()
        print(f'flow.load: {t1-t0:0.2f}, load_state_dict: {t2-t1:0.2f}')

    def batch_to_global(self, np_label, np_dense, np_sparse):
        labels = flow.tensor(np_label.reshape(-1, 1), dtype=flow.float)
        dense_fields = flow.tensor(np_dense, dtype=flow.float)
        sparse_fields = flow.tensor(np_sparse, dtype=flow.int64)
        labels = labels.to_global(placement=self.placement, sbp=self.sbp)
        dense_fields = dense_fields.to_global(placement=self.placement, sbp=self.sbp)
        sparse_fields = sparse_fields.to_global(placement=self.placement, sbp=self.sbp)
        return labels, dense_fields, sparse_fields


    def inference(self, dense_fields, sparse_fields):
        with flow.no_grad():
            logits = self.dlrm_module(dense_fields, sparse_fields)
            return logits

    def test(self):
        if self.eval_batchs == 0:
            return
        self.dlrm_module.eval()
        labels = []
        preds = []
        eval_start_time = time.time()
        with make_criteo_dataloader(self.args, "val") as val_loader:
            num_eval_batches = 0
            for np_batch in val_loader:
                num_eval_batches += 1
                if num_eval_batches > self.eval_batchs and self.eval_batchs > 0:
                    break
                label, dense_fields, sparse_fields = self.batch_to_global(*np_batch)
                logits = self.inference(dense_fields, sparse_fields)
                pred = logits.sigmoid()
                labels.append(label.numpy())
                preds.append(pred.numpy())

        if self.rank == 0:
            labels = np.concatenate(labels, axis=0)
            preds = np.concatenate(preds, axis=0)
            eval_time = time.time() - eval_start_time
            auc_start_time = time.time()
            auc = roc_auc_score(labels, preds)
            auc_time = time.time() - auc_start_time

            host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
            stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
            device_mem_str = stream.read().split("\n")[self.rank + 1]

            strtime = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'Rank[{self.rank}], Iter {self.cur_iter}, AUC {auc:0.5f}, ' +
                  f'Eval_time {eval_time:0.2f} s, AUC_time {auc_time:0.2f} s, ' +
                  f'#Samples {labels.shape[0]}, ' +
                  f'GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, ' +
                  f'{strtime}')


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    assert args.embedding_type != "OneEmbedding", "Do not support OneEmbedding in Eager Mode!"

    tester = Tester(args)
    tester.test()

