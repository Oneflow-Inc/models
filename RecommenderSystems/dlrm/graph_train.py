import oneflow as flow
import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from config import get_args
from eager_train import Trainer


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, wdl_module, use_fp16=False):
        super(DLRMValGraph, self).__init__()
        self.module = wdl_module
        if use_fp16:
            self.config.enable_amp(True)

    def build(self, dense_fields, sparse_fields):
        predicts = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        return predicts.to("cpu")


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
        logits = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        loss = self.bce_loss(logits, labels.to("cuda"))
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")


class GraphTrainer(Trainer):
    def __init__(self, args):
        super(GraphTrainer, self).__init__(args)
        if args.loss_scale_policy == "static":
            grad_scaler = flow.amp.StaticGradScaler(1024)
        else:
            grad_scaler = flow.amp.GradScaler(
                init_scale=1073741824,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )

        self.eval_graph = DLRMValGraph(self.dlrm_module, args.use_fp16)
        self.train_graph = DLRMTrainGraph(
            self.dlrm_module, self.loss, self.opt,
            self.lr_scheduler, grad_scaler, args.use_fp16
        )

    def train_one_step(self, labels, dense_fields, sparse_fields):
        return self.train_graph(labels, dense_fields, sparse_fields)

    def inference(self, dense_fields, sparse_fields):
        return self.eval_graph(dense_fields, sparse_fields)


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    trainer = GraphTrainer(args)
    trainer.train()
