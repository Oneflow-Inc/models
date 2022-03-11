import oneflow as flow
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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.save_path = args.model_save_dir
        self.eval_interval = args.eval_interval
        self.eval_batchs = args.eval_batchs

        self.cur_iter = 0
        self.rank = flow.env.get_rank()
        self.placement = flow.env.all_device_placement("cuda")
        self.sbp = flow.sbp.split(0)

        self.dlrm_module = make_dlrm_module(args)
        self.dlrm_module.to_global(self.placement, flow.sbp.broadcast)
        self.dlrm_module.embedding.set_model_parallel(self.placement)

        if args.model_load_dir != "":
            print(f"Loading model from {args.model_load_dir}")
            state_dict = flow.load(args.model_load_dir, global_src_rank=0)
            if args.mlp_type == 'FusedMLP':
                o2n = {
                    'bottom_mlp.linear_layers.fc0.features.0.weight': 'bottom_mlp.linear_layers.weight_0',
                    'bottom_mlp.linear_layers.fc0.features.0.bias': 'bottom_mlp.linear_layers.bias_0',
                    'bottom_mlp.linear_layers.fc1.features.0.weight': 'bottom_mlp.linear_layers.weight_1',
                    'bottom_mlp.linear_layers.fc1.features.0.bias': 'bottom_mlp.linear_layers.bias_1',
                    'bottom_mlp.linear_layers.fc2.features.0.weight': 'bottom_mlp.linear_layers.weight_2',
                    'bottom_mlp.linear_layers.fc2.features.0.bias': 'bottom_mlp.linear_layers.bias_2',
                    'top_mlp.linear_layers.fc0.features.0.weight': 'top_mlp.linear_layers.weight_0',
                    'top_mlp.linear_layers.fc0.features.0.bias': 'top_mlp.linear_layers.bias_0',
                    'top_mlp.linear_layers.fc1.features.0.weight': 'top_mlp.linear_layers.weight_1',
                    'top_mlp.linear_layers.fc1.features.0.bias': 'top_mlp.linear_layers.bias_1',
                    'top_mlp.linear_layers.fc2.features.0.weight': 'top_mlp.linear_layers.weight_2',
                    'top_mlp.linear_layers.fc2.features.0.bias': 'top_mlp.linear_layers.bias_2',
                    'top_mlp.linear_layers.fc3.features.0.weight': 'top_mlp.linear_layers.weight_3',
                    'top_mlp.linear_layers.fc3.features.0.bias': 'top_mlp.linear_layers.bias_3',
                    'top_mlp.linear_layers.fc4.features.weight': 'top_mlp.linear_layers.weight_4',
                    'top_mlp.linear_layers.fc4.features.bias': 'top_mlp.linear_layers.bias_4',
                }
                for old, new in o2n.items():
                    state_dict[new] = state_dict.pop(old)
            self.dlrm_module.load_state_dict(state_dict, strict=False)
        if args.save_initial_model and args.model_save_dir != "":
            self.save_model("initial_checkpoint")

        # opt = flow.optim.Adam(
        self.opt = flow.optim.SGD(self.dlrm_module.parameters(), lr=args.learning_rate)
        self.lr_scheduler = make_lr_scheduler(args, self.opt)
        self.loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

    def save_model(self, subdir):
        if self.save_path is None or self.save_path == '':
            return
        save_path = os.path.join(self.save_path, subdir)
        if self.rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = self.dlrm_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    def batch_to_global(self, np_label, np_dense, np_sparse):
        labels = flow.tensor(np_label.reshape(-1, 1), dtype=flow.float)
        dense_fields = flow.tensor(np_dense, dtype=flow.float)
        sparse_fields = flow.tensor(np_sparse, dtype=flow.int64)
        labels = labels.to_global(placement=self.placement, sbp=self.sbp)
        dense_fields = dense_fields.to_global(placement=self.placement, sbp=self.sbp)
        sparse_fields = sparse_fields.to_global(placement=self.placement, sbp=self.sbp)
        return labels, dense_fields, sparse_fields

    def train_one_step(self, labels, dense_fields, sparse_fields):
        logits = self.dlrm_module(dense_fields, sparse_fields)
        loss = self.loss(logits, labels)
        loss = flow.mean(loss)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
        return loss

    def train(self):
        self.dlrm_module.train()
        last_iter, last_time = 0, time.time()
        with make_criteo_dataloader(self.args, "train") as loader:
            for _ in range(self.args.max_iter):
                self.cur_iter += 1
                labels, dense_fields, sparse_fields = self.batch_to_global(*next(loader))
                loss = self.train_one_step(labels, dense_fields, sparse_fields)
                if self.cur_iter % self.args.loss_print_every_n_iter == 0:
                    loss = loss.numpy()
                    if self.rank == 0:
                        latency_ms = 1000 * (time.time() - last_time) / (self.cur_iter - last_iter)
                        last_iter, last_time = self.cur_iter, time.time()
                        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f'Rank[{self.rank}], Iter {self.cur_iter}, Loss {loss:0.4f}, '+
                              f'Latency {latency_ms:0.3f} ms, {strtime}')

                if self.eval_interval > 0 and self.cur_iter % self.eval_interval == 0:
                    self.eval()
                    last_time = time.time()

        if self.args.eval_after_training:
            if self.eval_interval > 0 and self.cur_iter % self.eval_interval != 0:
                self.eval()

    def inference(self, dense_fields, sparse_fields):
        with flow.no_grad():
            logits = self.dlrm_module(dense_fields, sparse_fields)
            return logits

    def eval(self):
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
        
        auc = 0 # will be updated by rank 0 only
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
        if self.args.save_model_after_each_eval:
            self.save_model(f"iter_{self.cur_iter}_val_auc_{auc:0.5f}")

        self.dlrm_module.train()


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    assert args.embedding_type != "OneEmbedding", "Do not support OneEmbedding in Eager Mode!"

    trainer = Trainer(args)
    trainer.train()
