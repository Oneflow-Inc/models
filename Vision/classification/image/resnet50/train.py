import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import time

import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp

from config import get_args
from graph import make_train_graph, make_eval_graph
from models.resnet50 import resnet50, Bottleneck
from models.data import make_data_loader
from models.optimizer import make_optimizer
from models.optimizer import make_lr_scheduler
from models.optimizer import make_cross_entropy
from models.accuracy import Accuracy
import utils.logger as log
from utils.stat import CudaUtilMemStat


class Trainer(object):
    def __init__(self):
        args = get_args()
        for k, v in args.__dict__.items():
            setattr(self, k, v)

        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()

        self.cur_epoch = 0
        self.cur_iter = 0
        self.cur_batch = 0
        self.is_global = (self.world_size > 1 and not self.ddp) or self.graph
        self.is_train = False
        self.meter_lr = self.graph is False

        self.init_logger()

        flow.boxing.nccl.set_fusion_threshold_mbytes(self.nccl_fusion_threshold_mb)
        flow.boxing.nccl.set_fusion_max_ops_num(self.nccl_fusion_max_ops)
        if self.use_fp16 and self.num_nodes * self.num_devices_per_node > 1:
            flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

        self.model = resnet50(
            zero_init_residual=self.zero_init_residual,
            fuse_bn_relu=self.fuse_bn_relu,
            fuse_bn_add_relu=self.fuse_bn_add_relu,
            channel_last=self.channel_last,
        )
        self.init_model()
        self.cross_entropy = make_cross_entropy(args)

        self.train_data_loader = make_data_loader(
            args, "train", self.is_global, self.synthetic_data
        )
        self.val_data_loader = make_data_loader(
            args, "validation", self.is_global, self.synthetic_data
        )

        self.optimizer = make_optimizer(args, self.model)
        self.lr_scheduler = make_lr_scheduler(args, self.optimizer)
        self.acc = Accuracy()

        if self.graph:
            self.train_graph = make_train_graph(
                self.model,
                self.cross_entropy,
                self.train_data_loader,
                self.optimizer,
                self.lr_scheduler,
                return_pred_and_label=self.metric_train_acc,
            )
            self.eval_graph = make_eval_graph(self.model, self.val_data_loader)

        if self.gpu_stat_file is not None:
            self.gpu_stat = CudaUtilMemStat(
                f"rank{self.rank}_" + self.gpu_stat_file, only_ordinal=self.rank
            )
        else:
            self.gpu_stat = None

    def init_model(self):
        self.logger.print("***** Model Init *****", print_ranks=[0])
        start_t = time.perf_counter()

        if self.is_global:
            placement = flow.env.all_device_placement("cuda")
            self.model = self.model.to_global(
                placement=placement, sbp=flow.sbp.broadcast
            )
        else:
            self.model = self.model.to("cuda")

        if self.load_path is None:
            self.legacy_init_parameters()
        else:
            self.load_state_dict()

        if self.ddp:
            self.model = ddp(self.model)

        if self.save_init:
            self.save("init")

        end_t = time.perf_counter()
        self.logger.print(
            f"***** Model Init Finish, time escapled: {end_t - start_t:.5f} s *****",
            print_ranks=[0],
        )

    def legacy_init_parameters(self):
        if not self.legacy_init:
            return

        for m in self.model.modules():
            # NOTE(zwx): legacy BatchNorm initializer in Benchmark seems wrong, so don't follow it
            if isinstance(m, flow.nn.Conv2d):
                flow.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, flow.nn.Linear):
                flow.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                flow.nn.init.constant_(m.bias, 0)
            elif isinstance(m, flow.nn.BatchNorm2d):
                flow.nn.init.constant_(m.weight, 1)
                flow.nn.init.constant_(m.bias, 0)

        for m in self.model.modules():
            if isinstance(m, Bottleneck):
                flow.nn.init.constant_(m.bn3.weight, 0)

    def load_state_dict(self):
        self.logger.print(f"Loading model from {self.load_path}", print_ranks=[0])
        if self.is_global:
            state_dict = flow.load(self.load_path, global_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.load_path)
        else:
            return

        self.model.load_state_dict(state_dict)

    def init_logger(self):
        if self.metric_local:
            print_ranks = list(range(self.world_size))
        else:
            print_ranks = [0]

        self.logger = log.get_logger(self.rank, print_ranks)
        self.logger.register_metric("job", log.IterationMeter(), "[{}]")
        self.logger.register_metric("epoch", log.IterationMeter(), "epoch: {}/{}")
        self.logger.register_metric("iter", log.IterationMeter(), "iter: {}/{}")
        self.logger.register_metric("loss", log.AverageMeter(), "loss: {:.5f}", True)
        if self.meter_lr:
            self.logger.register_metric("lr", log.IterationMeter(), "lr: {:.6f}")
        self.logger.register_metric("top1", log.AverageMeter(), "top1: {:.5f}", True)
        time_meter_str = (
            "throughput: {:.2f}, timestamp: {:.6f}"
            if self.print_timestamp
            else "throughput: {:.2f}"
        )
        self.logger.register_metric(
            "time", log.TimeMeter(self.print_timestamp), time_meter_str, True
        )

    def meter(
        self,
        epoch_pg=None,
        iter_pg=None,
        loss=None,
        lr=None,
        top1=None,
        num_samples=1,
        do_print=False,
    ):
        self.logger.meter("job", "train" if self.is_train else "eval")
        self.logger.meter("epoch", epoch_pg or (self.cur_epoch, self.num_epochs))
        self.logger.meter("iter", iter_pg or (self.cur_iter, self.batches_per_epoch))
        if loss is not None:
            self.logger.meter("loss", loss)

        if lr is not None and self.meter_lr:
            self.logger.meter("lr", lr)

        if top1 is not None:
            self.logger.meter("top1", top1)

        self.logger.meter("time", num_samples)

        if do_print:
            self.logger.print_metrics()
            if self.gpu_stat is not None:
                self.gpu_stat.stat()

    def meter_train_iter(self, loss, top1):
        assert self.is_train is True
        lr = None
        if self.meter_lr:
            lr = self.optimizer.param_groups[0]["lr"]

        do_print = (
            self.cur_iter % self.print_interval == 0
            or self.cur_iter == self.batches_per_epoch
        )
        self.meter(
            loss=loss,
            lr=lr,
            top1=top1,
            num_samples=self.train_batch_size,
            do_print=do_print,
        )

    def __call__(self):
        self.train()

    def train(self):
        self.logger.metric("time").reset()
        for _ in range(self.num_epochs):
            self.train_one_epoch()
            if self.cur_batch == self.total_batches:
                break

            if not self.skip_eval:
                acc = self.eval()
            else:
                acc = 0

            save_dir = f"epoch_{self.cur_epoch}_val_acc_{acc}"
            self.save(save_dir)
            self.cur_epoch += 1
            self.cur_iter = 0

    def train_one_epoch(self):
        self.model.train()
        self.is_train = True

        for _ in range(self.batches_per_epoch):
            if self.graph:
                loss, pred, label = self.train_graph()
            else:
                loss, pred, label = self.train_eager()

            self.cur_iter += 1

            loss = tol(loss, self.metric_local)
            if pred is not None and label is not None:
                pred = tol(pred, self.metric_local)
                label = tol(label, self.metric_local)
                top1_acc = self.acc([pred], [label])
            else:
                top1_acc = 0

            self.meter_train_iter(loss, top1_acc)

            self.cur_batch += 1
            if self.cur_batch == self.total_batches:
                break

    def train_eager(self):
        loss, pred, label = self.forward()

        if loss.is_global and self.scale_grad:
            # NOTE(zwx): scale init grad with world_size
            # because global_tensor.mean() include dividor numel * world_size
            loss = loss / self.world_size
            loss.backward()
            for param_group in self.optimizer.param_groups:
                for param in param_group.parameters:
                    param.grad /= self.world_size
        else:
            loss.backward()
            loss = loss / self.world_size

        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss, pred, label

    def eval(self):
        self.model.eval()
        self.is_train = False

        preds, labels = [], []
        for _ in range(self.val_batches_per_epoch):
            if self.graph:
                pred, label = self.eval_graph()
            else:
                pred, label = self.inference()

            preds.append(tton(pred, self.metric_local))
            labels.append(tton(label, self.metric_local))

        top1_acc = calc_acc(preds, labels)
        self.meter(
            iter_pg=(self.val_batches_per_epoch, self.val_batches_per_epoch),
            loss=0.0,
            top1=top1_acc,
            num_samples=self.val_batch_size * self.val_batches_per_epoch,
            do_print=True,
        )
        return top1_acc

    def forward(self):
        image, label = self.train_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        if self.metric_train_acc:
            pred = logits.softmax()
            return loss, pred, label
        else:
            return loss, None, None

    def inference(self):
        image, label = self.val_data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        with flow.no_grad():
            logits = self.model(image)
            pred = logits.softmax()

        return pred, label

    def save(self, subdir):
        if self.save_path is None:
            return

        save_path = os.path.join(self.save_path, subdir)
        self.logger.print(f"Saving model to {save_path}", print_ranks=[0])
        state_dict = self.model.state_dict()

        if self.is_global:
            flow.save(state_dict, save_path, global_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return


def tol(tensor, pure_local=True):
    """ to local """
    if tensor.is_global:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_global(sbp=flow.sbp.broadcast).to_local()

    return tensor


def tton(tensor, local_only=True):
    """ tensor to numpy """
    if tensor.is_global:
        if local_only:
            tensor = tensor.to_local().numpy()
        else:
            tensor = tensor.to_global(sbp=flow.sbp.broadcast).to_local().numpy()
    else:
        tensor = tensor.numpy()

    return tensor


def calc_acc(preds, labels):
    correct_of = 0.0
    num_samples = 0
    for pred, label in zip(preds, labels):
        clsidxs = np.argmax(pred, axis=1)
        correct_of += (clsidxs == label).sum()
        num_samples += label.size

    top1_acc = correct_of / num_samples
    return top1_acc


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
