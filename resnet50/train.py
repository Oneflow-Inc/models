import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import time
import math

import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp

from config import get_args
from models.resnet50 import resnet50
from models.data import make_data_loader
from models.optimizer import make_optimizer
from models.optimizer import make_grad_scaler
from models.optimizer import make_lr_scheduler
from models.optimizer import make_cross_entropy

# from utils.debug import dump_to_npy
# from utils.ofrecord_data_utils import OFRecordDataLoader


class Trainer(object):
    def __init__(self):
        args = get_args()

        self.cur_epoch_ = 0
        self.cur_iter_ = 0
        self.num_epochs_ = args.num_epochs
        self.batches_per_epoch_ = args.batches_per_epoch
        self.load_path_ = args.load

        self.with_graph_ = args.graph
        self.with_ddp_ = args.ddp

        if args.use_fp16 and not self.with_graph_:
            raise ValueError("NOT support fp16 in eager mode")

        self.rank_ = flow.distributed.get_rank()

        self.model = resnet50()
        self._init_model()

        self.train_data_loader = make_data_loader(args, "train")
        self.val_data_loader = make_data_loader(args, "validation")

        self.cross_entropy = make_cross_entropy(args)
        self.optimizer = make_optimizer(args, self.model)
        self.lr_scheduler = make_lr_scheduler(args, self.optimizer)

        self.model.to("cuda")
        self.cross_entropy.to("cuda")

    def _init_model(self):
        if self.load_path_ is None:
            pass
        else:
            if self.with_ddp_:
                if self.rank_ == 0:
                    self.model.load_state_dict(flow.load(self.load_path_))

                self.model = ddp(self.model)
            else:
                self.model.load_state_dict(
                    flow.load(self.load_path_, consistent_src_rank=0)
                )

    def __call__(self):
        if self.with_graph_:
            self._train_with_graph()
        else:
            self._train()

    def _train(self):
        for _ in range(self.num_epochs_):
            self._train_one_epoch()
            print("per epoch info")
            self.cur_epoch_ += 1
            self.cur_iter_ = 0

    def _train_one_epoch(self):
        for iter in range(self.batches_per_epoch_):
            loss = self._train_one_iter()
            print("per iter metric", loss)
            self.cur_iter_ += 1

    def _train_one_iter(self):
        image, label = self.train_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")

        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss


def main():
    args = get_args()

    flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)
    if args.use_fp16 and args.num_nodes * args.num_devices_per_node > 1:
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

    trainer = Trainer()
    trainer()


if __name__ == "__main__":
    main()
