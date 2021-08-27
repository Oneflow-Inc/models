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
from graph import make_train_graph, make_eval_graph
from models.resnet50 import resnet50
from models.data import make_data_loader
from models.optimizer import make_optimizer
from models.optimizer import make_lr_scheduler
from models.optimizer import make_cross_entropy
from utils.printer import Printer


class Trainer(object):
    def __init__(self, print_ranks=[0]):
        args = get_args()

        self.cur_epoch_ = 0
        self.cur_iter_ = 0
        self.num_epochs_ = args.num_epochs
        self.batches_ = args.batches_per_epoch
        self.val_batches_ = args.val_batches_per_epoch
        self.load_path_ = args.load
        self.save_path_ = args.save
        self.skip_eval_ = args.skip_eval

        self.rank_ = flow.env.get_rank()
        self.world_size_ = flow.env.get_world_size()
        self.print_ranks_ = print_ranks

        self.with_graph_ = args.graph
        self.with_ddp_ = args.ddp
        self.is_consistent_ = (
            self.world_size_ > 1 and not self.with_ddp_
        ) or self.with_graph_

        if args.use_fp16 and not self.with_graph_:
            raise ValueError("NOT support fp16 in eager mode")

        flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
        flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)
        if args.use_fp16 and args.num_nodes * args.num_devices_per_node > 1:
            flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

        self.model = resnet50()
        self.init_model()
        self.cross_entropy = make_cross_entropy(args)

        self.train_data_loader = make_data_loader(args, "train", self.is_consistent_)
        self.val_data_loader = make_data_loader(args, "validation", self.is_consistent_)

        self.optimizer = make_optimizer(args, self.model)
        self.lr_scheduler = make_lr_scheduler(args, self.optimizer)

        self.metric = Metric(
            self.rank_, args.print_interval, print_format=args.print_format
        )

        if self.with_graph_:
            self.train_graph = make_train_graph(
                self.model,
                self.cross_entropy,
                self.train_data_loader,
                self.optimizer,
                self.lr_scheduler,
            )
            self.eval_graph = make_eval_graph(self.model, self.val_data_loader)

    def rprint(self, *args):
        if self.rank_ not in self.print_ranks_:
            return

        print(*args)

    def init_model(self):
        self.rprint("***** Model Init *****")
        start_t = time.perf_counter()

        if self.is_consistent_:
            placement = flow.placement("cuda", {0: range(self.world_size_)})
            self.model = self.model.to_consistent(
                placement=placement, sbp=flow.sbp.broadcast
            )
        else:
            self.model = self.model.to("cuda")

        if self.load_path_ is None:
            self.init_parameters()
        else:
            self.load_state_dict()

        if self.with_ddp_:
            self.model = ddp(self.model)

        end_t = time.perf_counter()
        self.rprint(
            f"***** Model Init Finish, time escapled: {end_t - start_t:.5f} s *****"
        )

    def init_parameters(self):
        # raise NotImplementedError
        pass

    def load_state_dict(self):
        self.rprint(f"Loading states from {self.load_path_}")
        if self.is_consistent_:
            state_dict = flow.load(self.load_path_, consistent_src_rank=0)
        elif self.rank_ == 0:
            state_dict = flow.load(self.load_path_)
        else:
            return

        self.model.load_state_dict(state_dict)

    def __call__(self):
        self.train()

    def train(self):
        for _ in range(self.num_epochs_):
            if self.with_graph_:
                self.train_one_epoch_with_graph()
            else:
                self.train_one_epoch()

            self.cur_epoch_ += 1
            self.cur_iter_ = 0

            if not self.skip_eval_:
                acc = self.eval()

            self.save(acc)

    def train_one_epoch_with_graph(self):
        for _ in range(self.batches_):
            loss = self.train_graph()
            self.cur_iter_ += 1

            loss_np = loss.to_local().numpy() if loss.is_consistent else loss.numpy()
            self.metric.step(
                rank=self.rank_,
                epoch=self.cur_epoch_,
                iter=self.cur_iter_,
                loss=loss_np.item(),
                top1=0.0,
                job="train",
            )

    def train_one_epoch(self):
        self.model.train()

        for _ in range(self.batches_):
            loss = self.forward()
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.cur_iter_ += 1

            loss_np = loss.to_local().numpy() if loss.is_consistent else loss.numpy()
            self.metric.step(
                rank=self.rank_,
                epoch=self.cur_epoch_,
                iter=self.cur_iter_,
                loss=loss_np.item(),
                top1=0.0,
                job="train",
            )

    def forward(self):
        image, label = self.train_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        return loss

    def inference(self):
        image, label = self.val_data_loader()
        with flow.no_grad():
            logits = self.model(image.to("cuda"))
            predictions = logits.softmax()

        return predictions, label

    def eval(self):
        self.model.eval()

        correct_of = 0.0
        num_samples = 0
        for _ in range(self.val_batches_):
            if self.with_graph_:
                pred, label = self.eval_graph()
            else:
                pred, label = self.inference()

            if pred.is_consistent:
                # pred = pred.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
                pred = pred.to_local().numpy()
            else:
                pred = pred.numpy()

            if label.is_consistent:
                # label = label.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
                label = label.to_local().numpy()
            else:
                label = label.numpy()

            clsidxs = np.argmax(pred, axis=1)
            correct_of += (clsidxs == label).sum()
            num_samples += label.size

        top1_acc = correct_of / num_samples

        self.metric.print(
            rank=self.rank_,
            epoch=self.cur_epoch_,
            iter=self.cur_iter_,
            loss=0.0,
            top1=top1_acc,
            job="eval",
        )
        return top1_acc

    def save(self, acc):
        if self.save_path_ is None:
            return

        self.rprint(f"Saving states to {self.save_path_}")
        acc = acc or 0
        save_path = os.path.join(
            self.save_path_, "epoch_{}_val_acc_{}".format(self.cur_epoch_, acc),
        )
        state_dict = self.model.state_dict()

        if self.is_consistent_:
            flow.save(state_dict, save_path, consistent_dst_rank=0)
        elif self.rank_ == 0:
            flow.save(state_dict, save_path)
        else:
            return


def _last(s):
    return s.iloc[-1]


def _last_f(s):
    return "{:.5f}".format(s.iloc[-1])


def _mean(s):
    return "{:.5f}".format(s.to_numpy().mean())


class Metric(object):
    def __init__(
        self, rank, print_interval, print_only_on_rank=None, print_format="normal"
    ):
        self.rank_ = rank
        self.print_interval_ = print_interval
        self.print_only_on_rank_ = print_only_on_rank
        self.step_ = 0

        self.p_ = Printer(
            ("rank", "epoch", "iter", "job", "loss", "top1"), print_format
        )

        self.p_.register_handler("rank", _last)
        self.p_.register_handler("epoch", _last)
        self.p_.register_handler("iter", _last)
        self.p_.register_handler("job", _last)
        self.p_.register_handler("loss", _mean)
        self.p_.register_handler("top1", _last_f)

        self.p_.register_str_len("rank", 5)
        self.p_.register_str_len("epoch", 5)
        self.p_.register_str_len("iter", 8)
        self.p_.register_str_len("job", 7)
        self.p_.register_str_len("loss", 10)
        self.p_.register_str_len("top1", 10)

        self.p_.finish()

    def skip(self):
        if (
            self.print_only_on_rank_ is not None
            and self.print_only_on_rank_ != self.rank_
        ):
            return True

        return False

    def step(self, **kwargs):
        if self.skip():
            return

        self.p_.record(**kwargs)

        self.step_ += 1
        if self.step_ == self.print_interval_:
            self.p_.print()
            self.step_ = 0

    def print(self, **kwargs):
        if self.skip():
            return

        self.p_.record(**kwargs)
        self.p_.reset_title_printed()
        self.p_.print()


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
