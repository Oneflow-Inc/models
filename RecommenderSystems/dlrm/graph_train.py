import oneflow as flow
import os
import sys
import glob
import time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from config import get_args
from crieto_dataset_util import CrietoDatasetContextManager
from dlrm import make_dlrm_module
from lr_scheduler import make_lr_scheduler


def make_crieto_dataloader(args, mode):
    """Make a Crieto Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader
                will be closed.
    """
    subfolders = args.train_sub_folders if mode=='train' else args.val_sub_folders

    files = []
    for folder in subfolders:
        files += ['file://' + name for name in glob.glob(f'{args.data_dir}/{folder}/*.parquet')]
    files.sort()

    return CrietoDatasetContextManager(
        files,
        args.batch_size_per_proc if mode=='train' else args.eval_batch_size_per_proc,
        None if mode=='train' else 1,
        num_dense_fields=args.num_dense_fields,
        num_sparse_fields=args.num_sparse_fields,
        shuffling_queue_capacity=(mode=='train'),
        shard_seed=1234,
        shard_count=flow.env.get_world_size(),
        cur_shard=flow.env.get_rank())


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, use_fp16=False):
        super(DLRMValGraph, self).__init__()
        self.module = wdl_module
        if use_fp16:
            self.config.enable_amp(True)

    def build(self, labels, dense_fields, sparse_fields):
        predicts = self.module(dense_fields, sparse_fields)
        return predicts, labels


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(self, wdl_module, bce_loss, optimizer, lr_scheduler=None, grad_scaler=None, use_fp16=False):
        super(DLRMTrainGraph, self).__init__()
        self.module = wdl_module
        self.bce_loss = bce_loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if use_fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, dense_fields, sparse_fields):
        logits = self.module(dense_fields, sparse_fields)
        loss = self.bce_loss(logits, labels)
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss


class Trainer(object):
    def __init__(self):
        args = get_args()
        self.args = args
        self.save_path = args.model_save_dir
        self.save_model_after_each_eval = args.save_model_after_each_eval
        self.eval_after_training = args.eval_after_training
        self.max_iter = args.max_iter
        self.loss_print_every_n_iter = args.loss_print_every_n_iter
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
            print(f"Loading model from {self.args.model_load_dir}")
            state_dict = flow.load(self.args.model_load_dir, global_src_rank=0)
            self.dlrm_module.load_state_dict(state_dict, strict=False)
        if args.save_initial_model and args.model_save_dir != "":
            self.save_model("initial_checkpoint")

        # opt = flow.optim.Adam(
        opt = flow.optim.SGD(self.dlrm_module.parameters(), lr=args.learning_rate)
        lr_scheduler = make_lr_scheduler(args, opt)
        if args.loss_scale_policy == "static":
            grad_scaler = flow.amp.StaticGradScaler(1024)
        else:
            grad_scaler = flow.amp.GradScaler(
                init_scale=1073741824,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )

        self.loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")
        self.eval_graph = DLRMValGraph(self.dlrm_module, args.use_fp16)
        self.train_graph = DLRMTrainGraph(
            self.dlrm_module, self.loss, opt,
            lr_scheduler, grad_scaler, args.use_fp16
        )

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

    def train(self):
        self.dlrm_module.train()
        last_iter, last_time = 0, time.time()
        with make_crieto_dataloader(self.args, "train") as train_loader:
            for _ in range(self.max_iter):
                self.cur_iter += 1
                labels, dense_fields, sparse_fields = self.batch_to_global(*next(train_loader))
                loss = self.train_graph(labels, dense_fields, sparse_fields)
                if self.cur_iter % self.loss_print_every_n_iter == 0:
                    loss = loss.numpy()
                    if self.rank == 0:
                        latency_ms = 1000 * (time.time() - last_time) / (self.cur_iter - last_iter)
                        last_iter, last_time = self.cur_iter, time.time()
                        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f'Iter {self.cur_iter}, Loss {loss:0.4f}, Latency_ms {latency_ms:0.3f}, {strtime}')

                if self.eval_interval > 0 and self.cur_iter % self.eval_interval == 0:
                    self.eval()
                    last_time = time.time()

        if self.eval_after_training:
            if self.eval_interval > 0 and self.cur_iter % self.eval_interval != 0:
                self.eval()

    def eval(self):
        if self.eval_batchs == 0:
            return
        self.dlrm_module.eval()
        labels = []
        preds = []
        with make_crieto_dataloader(self.args, "val") as val_loader:
            num_eval_batches = 0
            for np_batch in val_loader:
                num_eval_batches += 1
                if num_eval_batches > self.eval_batchs and self.eval_batchs > 0:
                    break
                label, dense_fields, sparse_fields = self.batch_to_global(*np_batch)
                logits, label =  self.eval_graph(label, dense_fields, sparse_fields)
                pred = logits.sigmoid()
                labels.append(label.numpy())
                preds.append(pred.numpy())
        
        if self.rank == 0:
            labels = np.concatenate(labels, axis=0)
            preds = np.concatenate(preds, axis=0)
            auc = roc_auc_score(labels, preds)
        
            strtime = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'Iter {self.cur_iter}, AUC {auc:0.5f}, #Samples {labels.shape[0]}, {strtime}')
            if self.save_model_after_each_eval:
                self.save_model(f"iter_{self.cur_iter}_val_auc_{auc}")
        self.dlrm_module.train()


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    trainer = Trainer()
    trainer.train()
