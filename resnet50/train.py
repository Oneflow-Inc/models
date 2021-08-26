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
from utils.printer import Printer

# from utils.debug import dump_to_npy
# from utils.ofrecord_data_utils import OFRecordDataLoader


class Trainer(object):
    def __init__(self):
        args = get_args()

        self.cur_epoch_ = 0
        self.cur_iter_ = 0
        self.num_epochs_ = args.num_epochs
        self.batches_ = args.batches_per_epoch
        self.val_batches_ = args.val_batches_per_epoch
        self.load_path_ = args.load

        self.with_graph_ = args.graph
        self.with_ddp_ = args.ddp

        if args.use_fp16 and not self.with_graph_:
            raise ValueError("NOT support fp16 in eager mode")

        self.rank_ = flow.distributed.get_rank()

        self.model = resnet50()
        self._init_model()
        self.cross_entropy = make_cross_entropy(args)

        self.model.to("cuda")
        self.cross_entropy.to("cuda")

        self.train_data_loader = make_data_loader(args, "train")
        self.val_data_loader = make_data_loader(args, "validation")

        self.optimizer = make_optimizer(args, self.model)
        self.lr_scheduler = make_lr_scheduler(args, self.optimizer)

        self.metric = Metric(self.rank_, args.print_interval)

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
            self.cur_epoch_ += 1
            self.cur_iter_ = 0
            self._eval()

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        for _ in range(self.batches_):
            loss = self._forward()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.cur_iter_ += 1

            loss_np = loss.to_local().numpy() if loss.is_consistent else loss.numpy()
            self.metric.step(
                epoch=self.cur_epoch_,
                iter=self.cur_iter_,
                loss=loss_np.item(),
                top1=0.0,
                job="train",
            )

    def _forward(self):
        image, label = self.train_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")

        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        return loss

    def _inference(self):
        image, label = self.val_data_loader()
        logits = self.model(image.to("cuda"))
        with flow.no_grad():
            predictions = logits.softmax()

        return predictions, label

    def _eval(self):
        self.model.eval()

        correct_of = 0.0
        num_samples = 0
        for _ in range(self.val_batches_):
            pred, label = self._inference()

            if pred.is_consistent:
                pred = pred.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
            else:
                pred = pred.numpy()

            if label.is_consistent:
                label = label.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
            else:
                label = label.numpy()

            clsidxs = np.argmax(pred, axis=1)
            correct_of += (clsidxs == label).sum()
            num_samples += label.size

        top1_acc = correct_of / num_samples

        self.metric.print(
            epoch=self.cur_epoch_,
            iter=self.cur_iter_,
            loss=0.0,
            top1=top1_acc,
            job="eval",
        )


def _last(s):
    return s.iloc[-1]


def _last_f(s):
    return "{:.5f}".format(s.iloc[-1])


def _mean(s):
    return "{:.5f}".format(s.to_numpy().mean())


class Metric(object):
    def __init__(self, rank, print_interval):
        self.rank_ = rank
        self.print_interval_ = print_interval
        self.step_ = 0

        if self.rank_ == 0:
            self.p_ = Printer(("epoch", "iter", "job", "loss", "top1"))

            self.p_.register_handler("epoch", _last)
            self.p_.register_handler("iter", _last)
            self.p_.register_handler("job", _last)
            self.p_.register_handler("loss", _mean)
            self.p_.register_handler("top1", _last_f)

            self.p_.register_str_len("epoch", 5)
            self.p_.register_str_len("iter", 8)
            self.p_.register_str_len("job", 7)
            self.p_.register_str_len("loss", 10)
            self.p_.register_str_len("top1", 10)

            self.p_.finish()

    def step(self, **kwargs):
        if self.rank_ != 0:
            return

        self.p_.record(**kwargs)

        self.step_ += 1
        if self.step_ == self.print_interval_:
            self.p_.print()
            self.step_ = 0

    def print(self, **kwargs):
        if self.rank_ != 0:
            return

        self.p_.record(**kwargs)
        self.p_.print()


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
