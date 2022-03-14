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
from eager_train import make_criteo_dataloader


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
        with make_criteo_dataloader(self.args, "test") as val_loader:
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

