import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import oneflow as flow
import oneflow._oneflow_internal

from oneflow_gpt.config import get_args
from oneflow_gpt import distribute as dist
from oneflow_gpt.data import GPTDataLoader
from oneflow_gpt.model import GPTModel, Embedding, Logits
from oneflow_gpt.model import Transformer, TransformerLayer, ActivationCheckpointing
from oneflow_gpt.model import ParallelSparseSoftmaxCrossEntropyLoss
from oneflow_gpt.optimizer import make_optimizer, make_lr_scheduler, make_grad_scaler
from oneflow_gpt.logger import print_rank_0, print_rank_last, Logger
from oneflow_gpt.logger import IterationMetric, AccumulationMetric
from oneflow_gpt.logger import LossMetric, ThroughputMetric


class Trainer(object):
    def __init__(self):
        self.args = get_args()
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.model = GPTModel()
        self.data_loader = GPTDataLoader()
        self.cross_entropy = ParallelSparseSoftmaxCrossEntropyLoss()
        self.optimizer = make_optimizer(self.args, self.model)
        self.lr_scheduler = make_lr_scheduler(self.args, self.optimizer)
        # self.optimizer = None
        # self.lr_scheduler = None
        # NOTE(zwx): grad scaler is not available in eager mode
        self.grad_scaler = make_grad_scaler(self.args)

        if self.args.graph:
            flow.boxing.nccl.enable_use_compute_stream()

            self.train_graph = GPTGraph(
                self.model,
                self.data_loader,
                self.cross_entropy,
                self.optimizer,
                self.lr_scheduler,
                self.grad_scaler,
            )

        # self.save("init")

        self.logger = Logger(self.rank)
        self.logger.register_metric("iter", IterationMetric())
        self.logger.register_metric("samples", AccumulationMetric())
        self.logger.register_metric("loss", LossMetric(), "loss: {:.5f}", True)
        self.logger.register_metric(
            "throughput", ThroughputMetric(), "throughput: {:.2f}", True
        )

    def __call__(self):
        iteration = 0
        while iteration < self.args.train_iters:
            if self.args.graph:
                loss = self.train_graph()
            else:
                raise NotImplementedError
                # loss = self.train_eager()

            if loss.is_consistent:
                loss = loss.to_local()

            # snapshot.step()
            # iteration = snapshot.iter
            iteration += 1
            print(f"[{self.rank}] iteration: {iteration}")

            self.logger.meter("samples", self.args.global_batch_size)
            self.logger.meter("loss", loss)
            self.logger.meter("throughput", self.args.global_batch_size)
            if iteration % self.args.log_interval == 0:
                self.logger.meter("iter", iteration)
                self.logger.print_metrics([self.world_size - 1])

        print(f"[{self.rank}] training finished")
        # oneflow._oneflow_internal.eager.multi_client.Sync()

    def train_eager(self):
        data, label = self.data_loader()
        logits = self.model(data)
        loss = self.cross_entropy(logits, label)
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss

    def save(self, subdir):
        if self.args.checkpoint_save_path is None:
            return

        save_path = os.path.join(self.args.checkpoint_save_path, subdir)
        print_rank_0(f"Saving model to {save_path}")
        state_dict = self.model.state_dict()

        flow.save(state_dict, save_path, consistent_dst_rank=0)


class GPTGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        data_loader,
        cross_entropy,
        optimizer=None,
        lr_scheduler=None,
        grad_scaler=None,
    ):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.cross_entropy = cross_entropy
        self.is_train = False
        if optimizer is not None:
            self.is_train = True
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            if grad_scaler is not None:
                self.set_grad_scaler(grad_scaler)

        args = get_args()
        self.set_activation_checkpointing()
        self.set_pipeline_stage_id()
        self.config.set_gradient_accumulation_steps(args.num_accumulation_steps)

        if args.fp16:
            self.config.enable_amp(True)

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_cast_scale(True)

    def __del__(self):
        print(f"[{flow.env.get_rank()}] graph del")

    def set_activation_checkpointing(self):
        for module_block in self.model.modules():
            if isinstance(module_block.origin, TransformerLayer):
                module_block.config.activation_checkpointing = True

    def set_pipeline_stage_id(self):
        dist_util = dist.get_dist_util()

        self.data_loader.config.stage_id = dist_util.get_layer_stage_id(0)
        self.data_loader.data_decoder.config.stage_id = dist_util.get_layer_stage_id(0)

        for module_block in self.model.modules():
            if isinstance(module_block.origin, Embedding):
                module_block.config.stage_id = dist_util.get_layer_stage_id(0)
            elif isinstance(
                module_block.origin, (TransformerLayer, ActivationCheckpointing)
            ):
                module_block.config.stage_id = dist_util.get_layer_stage_id(
                    module_block.origin.layer_idx
                )
            elif isinstance(module_block.origin, Transformer):
                module_block.config.stage_id = dist_util.get_layer_stage_id(-1)
            elif isinstance(module_block.origin, Logits):
                module_block.config.stage_id = dist_util.get_layer_stage_id(-1)
            else:
                pass

        self.data_loader.label_decoder.config.stage_id = dist_util.get_layer_stage_id(
            -1
        )
        self.cross_entropy.config.stage_id = dist_util.get_layer_stage_id(-1)

    def build(self):
        data, label = self.data_loader()
        logits = self.model(data)
        loss = self.cross_entropy(logits, label)
        if self.is_train:
            loss.backward()
        return loss


if __name__ == "__main__":
    Trainer()()
    print(f"[{flow.env.get_rank()}] exit")
