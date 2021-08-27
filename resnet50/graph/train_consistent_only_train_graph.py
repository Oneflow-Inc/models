
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse
import numpy as np
import time
import math

import oneflow as flow
from oneflow.nn.module import Module

# from utils.debug import dump_to_npy
from utils.ofrecord_data_utils import OFRecordDataLoader
from models.resnet50 import resnet50

_GLOBAL_VARS = None


def parse_args():
    def str_list(x):
        return [i.strip() for i in x.split(",")]

    def int_list(x):
        return list(map(int, x.split(",")))

    def float_list(x):
        return list(map(float, x.split(",")))

    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("flags for train resnet50")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path", type=str, default="./ofrecord", help="dataset path"
    )
    parser.add_argument(
        "--ofrecord_part_num", type=int, default=1, help="ofrecord data part number"
    )
    # training hyper-parameters
    parser.add_argument(
        "--train_batch_size_per_device", type=int, default=32, help="train batch size"
    )
    parser.add_argument("--val_batch_size_per_device", type=int, default=32, help="val batch size")
    parser.add_argument(
        "--process_num_per_node", type=int, default=1, help=""
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1, help="node/machine number for training"
    )
    parser.add_argument("--learning_rate", type=float, default=0.256)
    parser.add_argument("--wd", type=float, default=1.0 / 32768, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.875, help="momentum")
    parser.add_argument(
        "--lr_decay",
        type=str,
        default="cosine",
        help="cosine, step, polynomial, exponential, None",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="the epochs to warmp-up lr to scaled large-batch value",
    )
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
    )
    parser.add_argument("--num_epochs", type=int, default=90, help="number of epochs")

    parser.add_argument(
        "--nccl_fusion_threshold_mb",
        type=int,
        default=0,
        help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--nccl_fusion_max_ops",
        type=int,
        default=0,
        help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.",
    )

    # for data process
    parser.add_argument(
        "--num_classes", type=int, default=1000, help="num of pic classes"
    )
    parser.add_argument(
        "--train_examples_num", type=int, default=1281167, help="train pic number"
    )
    parser.add_argument(
        "--val_examples_num", type=int, default=50000, help="validation pic number"
    )
    parser.add_argument(
        "--rgb-mean",
        type=float_list,
        default=[123.68, 116.779, 103.939],
        help="a tuple of size 3 for the mean rgb",
    )
    parser.add_argument(
        "--rgb-std",
        type=float_list,
        default=[58.393, 57.12, 57.375],
        help="a tuple of size 3 for the std rgb",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1, help="label smoothing factor"
    )

    # log and loss print
    parser.add_argument(
        "--loss_print_every_n_iter",
        type=int,
        default=100,
        help="print loss every n iteration",
    )
    return parser.parse_args()


class GlobalVars(object):
    def __init__(self, args):
        self.rank = flow.distributed.get_rank()
        self.world_size = flow.distributed.get_world_size()
        self.total_train_batch_size = args.train_batch_size_per_device * self.world_size
        self.total_val_batch_size = args.val_batch_size_per_device * self.world_size
        self.batches_per_epoch = math.ceil(args.train_examples_num / self.total_train_batch_size)
        self.val_batches_per_epoch = int(args.val_examples_num / self.total_val_batch_size)
        self.warmup_batches = self.batches_per_epoch * args.warmup_epochs
        self.num_train_batches = self.batches_per_epoch * args.num_epochs
        self.decay_batches = self.num_train_batches - self.warmup_batches  # TODO: remove warmup_batches


def global_vars():
    global _GLOBAL_VARS
    assert _GLOBAL_VARS is not None
    return _GLOBAL_VARS


class LabelSmoothLoss(Module):
    def __init__(self, num_classes=-1, smooth_rate=0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth_rate = smooth_rate
        self.on_value = 1 - self.smooth_rate + self.smooth_rate / self.num_classes
        self.off_value = self.smooth_rate / self.num_classes

    def forward(self, input, label):
        onehot_label = flow.F.one_hot(label, self.num_classes, self.on_value, self.off_value)
        loss = flow.F.softmax_cross_entropy(input, onehot_label.to(dtype=input.dtype))
        return loss.mean()


def prepare_modules(args, to_consistent=True):
    vars = global_vars()

    device_list = [i for i in range(args.process_num_per_node)]
    placement = flow.placement("cpu", {0: device_list}) if to_consistent else None
    sbp = [flow.sbp.split(0)] if to_consistent else None

    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=args.train_examples_num,
        batch_size=vars.total_train_batch_size,
        total_batch_size=vars.total_train_batch_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    # oneflow init
    start_t = time.time()
    resnet50_module = resnet50()

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))
    of_cross_entropy = LabelSmoothLoss(num_classes=args.num_classes, smooth_rate=args.label_smoothing)

    def load_ckpt():
        if args.load_checkpoint != "":
            loaded_state_dict = flow.load(
                args.load_checkpoint, consistent_src_rank=0 if to_consistent else None
            )
            print("rank %d load_checkpoint >>>>>>>>> " % vars.rank, args.load_checkpoint)
            resnet50_module.load_state_dict(loaded_state_dict)

    if to_consistent:
        placement = flow.placement("cuda", {0: device_list})
        sbp = [flow.sbp.broadcast]
        resnet50_module.to_consistent(placement=placement, sbp=sbp)
        of_cross_entropy.to_consistent(placement=placement, sbp=sbp)
        load_ckpt()
    else:
        resnet50_module.to("cuda")
        of_cross_entropy.to("cuda")
        load_ckpt()

    print(args)

    optimizer = flow.optim.SGD(
        resnet50_module.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=vars.decay_batches
    )

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler, warmup_factor=0, warmup_iters=vars.warmup_batches, warmup_method="linear"
    )

    return train_data_loader, resnet50_module, of_cross_entropy, optimizer, lr_scheduler


def main(args):
    flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)
    if args.use_fp16 and args.num_nodes * args.process_num_per_node > 1:
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

    vars = global_vars()

    train_data_loader, resnet50_module, cross_entropy_loss, optimizer, lr_scheduler = prepare_modules(args)

    class Resnet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            if args.use_fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

            self.config.allow_fuse_add_to_output(True)
            self.config.allow_fuse_model_update_ops(True)

            self.train_data_loader = train_data_loader
            self.resnet50 = resnet50_module
            self.cross_entropy = cross_entropy_loss
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)

        def build(self):
            image, label = self.train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            logits = self.resnet50(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    resnet50_graph = Resnet50Graph()

    of_losses, of_accuracy = [], []
    for epoch in range(args.num_epochs):
        resnet50_module.train()

        for b in range(vars.batches_per_epoch):
            # oneflow graph train
            start_t = time.time()

            loss = resnet50_graph()

            end_t = time.time()

            if b % args.loss_print_every_n_iter == 0:
                correct_of = 0.0
                loss = loss.to_local()
                loss_np = loss.numpy()
                of_losses.append(loss_np)

                print(
                    "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}".format(
                        "train", epoch, b, loss_np, correct_of / vars.total_train_batch_size, -1, -1
                    )
                )

        # begin eval
        print("rank {} epoch {} train done, start validation".format(vars.rank, epoch))

        flow.save(
            resnet50_module.state_dict(),
            os.path.join(
                args.save_checkpoint_path,
                "epoch_%d" % epoch,
            ),
            consistent_dst_rank=0
        )

if __name__ == "__main__":
    args = parse_args()
    assert _GLOBAL_VARS is None
    _GLOBAL_VARS = GlobalVars(args)
    main(args)
