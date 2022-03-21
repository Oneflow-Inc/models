import oneflow as flow
import os
import sys
import glob
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import get_args
from dataloader import DLRMDataReader
from dlrm import make_dlrm_module


def make_criteo_dataloader(data_path, batch_size, shuffle=True):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return DLRMDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=1234,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=args.warmup_batches,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, steps=args.decay_batches, end_learning_rate=0, power=2.0, cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[args.decay_start],
        interval_rescaling=True,
    )
    return sequential_lr


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, dlrm_module, amp=False):
        super(DLRMValGraph, self).__init__()
        self.module = dlrm_module
        if amp:
            self.config.enable_amp(True)

    def build(self, dense_fields, sparse_fields):
        predicts = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        return predicts.to("cpu")


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(
        self, dlrm_module, loss, optimizer, lr_scheduler=None, grad_scaler=None, amp=False,
    ):
        super(DLRMTrainGraph, self).__init__()
        self.module = dlrm_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, dense_fields, sparse_fields):
        logits = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")


def train(args):
    rank = flow.env.get_rank()

    dlrm_module = make_dlrm_module(args)
    dlrm_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    if args.model_load_dir:
        print(f"Loading model from {args.model_load_dir}")
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        dlrm_module.load_state_dict(state_dict, strict=False)

    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = dlrm_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    opt = flow.optim.SGD(dlrm_module.parameters(), lr=args.learning_rate)
    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DLRMValGraph(dlrm_module, args.amp)
    train_graph = DLRMTrainGraph(dlrm_module, loss, opt, lr_scheduler, grad_scaler, args.amp)
    dlrm_module.train()
    last_iter, last_time = 0, time.time()
    with make_criteo_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        for iter in range(1, args.max_iter + 1):
            labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            loss = train_graph(labels, dense_fields, sparse_fields)
            if iter % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency_ms = 1000 * (time.time() - last_time) / (iter - last_iter)
                    last_iter, last_time = iter, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Iter {iter}, Loss {loss:0.4f}, "
                        + f"Latency {latency_ms:0.3f} ms, {strtime}"
                    )

            if args.eval_interval > 0 and iter % args.eval_interval == 0:
                auc = eval(args, eval_graph, iter)
                if args.save_model_after_each_eval:
                    save_model(f"iter_{iter}_val_auc_{auc:0.5f}")
                dlrm_module.train()
                last_time = time.time()

    if args.eval_interval > 0 and iter % args.eval_interval != 0:
        auc = eval(args, eval_graph, iter)
        if args.save_model_after_each_eval:
            save_model(f"iter_{iter}_val_auc_{auc:0.5f}")


def batch_to_global(np_label, np_dense, np_sparse):
    def _np_to_global(np, dtype=flow.float):
        t = flow.tensor(np, dtype=dtype)
        return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))

    labels = _np_to_global(np_label.reshape(-1, 1))
    dense_fields = _np_to_global(np_dense)
    sparse_fields = _np_to_global(np_sparse, dtype=flow.int64)
    return labels, dense_fields, sparse_fields


def eval(args, eval_graph, cur_iter=0):
    if args.eval_batches <= 0:
        return
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    with make_criteo_dataloader(
        f"{args.data_dir}/test", args.eval_batch_size, shuffle=False
    ) as loader:
        num_eval_batches = 0
        for np_batch in loader:
            num_eval_batches += 1
            if num_eval_batches > args.eval_batches:
                break
            label, dense_fields, sparse_fields = batch_to_global(*np_batch)
            logits = eval_graph(dense_fields, sparse_fields)
            pred = logits.sigmoid()
            labels.append(label.numpy())
            preds.append(pred.numpy())

    auc = 0  # will be updated by rank 0 only
    rank = flow.env.get_rank()
    if rank == 0:
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        eval_time = time.time() - eval_start_time
        auc_start_time = time.time()
        auc = roc_auc_score(labels, preds)
        auc_time = time.time() - auc_start_time

        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Iter {cur_iter}, AUC {auc:0.5f}, Eval_time {eval_time:0.2f} s, "
            + f"AUC_time {auc_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    flow.comm.barrier()
    return auc


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()

    train(args)
