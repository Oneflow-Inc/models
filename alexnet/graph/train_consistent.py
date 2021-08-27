
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

from utils.debug import dump_to_npy
from utils.ofrecord_data_utils import OFRecordDataLoader
from model.alexnet import alexnet

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

    parser = argparse.ArgumentParser("flags for train alexnet")
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
        self.batches_per_epoch = math.ceil(args.train_examples_num // self.total_train_batch_size)
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
        # log_prob = input.softmax(dim=-1).log()
        # onehot_label = flow.F.cast(onehot_label, log_prob.dtype)
        # loss = flow.mul(log_prob * -1, onehot_label).sum(dim=-1).mean()
        loss = flow.F.softmax_cross_entropy(input, onehot_label.to(dtype=input.dtype))
        return loss.mean()


def prepare_modules(args, to_consistent=True):
    vars = global_vars()

    device_list = [i for i in range(args.process_num_per_node)]
    placement = flow.placement("cpu", {0: device_list}) if to_consistent else None
    sbp = [flow.sbp.split(0)] if to_consistent else None

    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="validation",  # "train",
        dataset_size=args.train_examples_num,
        batch_size=vars.total_train_batch_size,
        total_batch_size=vars.total_train_batch_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="validation",
        dataset_size=args.val_examples_num,
        batch_size=vars.total_val_batch_size,
        total_batch_size=vars.total_val_batch_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    # oneflow init
    start_t = time.time()
    alexnet_module = alexnet()

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    if args.label_smoothing > 0:
        of_cross_entropy = LabelSmoothLoss(num_classes=args.num_classes, smooth_rate=args.label_smoothing)
    else:
        of_cross_entropy = flow.nn.CrossEntropyLoss(reduction='mean')

    def load_ckpt():
        if args.load_checkpoint != "":
            loaded_state_dict = flow.load(
                args.load_checkpoint, consistent_src_rank=0 if to_consistent else None
            )
            print("rank %d load_checkpoint >>>>>>>>> " % vars.rank, args.load_checkpoint)
            alexnet_module.load_state_dict(loaded_state_dict)

    if to_consistent:
        placement = flow.placement("cuda", {0: device_list})
        sbp = [flow.sbp.broadcast]
        alexnet_module.to_consistent(placement=placement, sbp=sbp)
        of_cross_entropy.to_consistent(placement=placement, sbp=sbp)
        load_ckpt()
    else:
        alexnet_module.to("cuda")
        of_cross_entropy.to("cuda")
        load_ckpt()

    print(args)

    # flow.save(alexnet_module.state_dict(), "init_ckpt", consistent_dst_rank=0)
    # exit()

    # print('named_parameters', '*'*100)
    # for name, param in alexnet_module.named_parameters():
    #     print(name)
    # print('named_buffers', '*'*100)
    # for name, param in alexnet_module.named_buffers():
    #     print(name)
    # print('*'*100)
    # exit()

    optimizer = flow.optim.SGD(
        alexnet_module.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    lr_scheduler = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=vars.decay_batches
    )

    if args.warmup_epochs > 0:
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
            lr_scheduler, warmup_factor=0, warmup_iters=vars.warmup_batches, warmup_method="linear"
        )

    return train_data_loader, val_data_loader, alexnet_module, of_cross_entropy, optimizer, lr_scheduler


def main(args):
    flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)
    if args.use_fp16 and args.num_nodes * args.process_num_per_node > 1:
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

    vars = global_vars()

    train_data_loader, val_data_loader, alexnet_module, cross_entropy_loss, optimizer, lr_scheduler = prepare_modules(args)

    class AlexNetGraph(flow.nn.Graph):
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
            self.alexnet = alexnet_module
            self.cross_entropy = cross_entropy_loss
            self.add_optimizer(optimizer)
            # self.add_optimizer(optimizer, lr_sch=lr_scheduler)

        def build(self):
            image, label = self.train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            logits = self.alexnet(image)
            loss = self.cross_entropy(logits, label)
            predictions = logits.softmax()
            loss.backward()
            return loss, predictions, image, label, logits

    alexnet_graph = AlexNetGraph()

    class AlexNetEvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.val_data_loader = val_data_loader
            self.alexnet = alexnet_module

            if args.use_fp16:
                self.config.enable_amp(True)

            self.config.allow_fuse_add_to_output(True)

        def build(self):
            image, label = self.val_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            with flow.no_grad():
                logits = self.alexnet(image)
                predictions = logits.softmax()
            return predictions, label

    alexnet_eval_graph = AlexNetEvalGraph()

    if vars.rank == 0:
        writer_train_loss = open("graph/train_losses.txt", "w")
        writer_train_top1_acc = open("graph/train_top1_accuracy.txt", "w")
        writer_val_top1_acc = open("graph/val_top1_accuracy.txt", "w")

    of_losses, of_accuracy = [], []
    for epoch in range(args.num_epochs):
        alexnet_module.train()

        for b in range(vars.batches_per_epoch):
            # oneflow graph train
            start_t = time.time()

            loss, predictions, images, label, logits = alexnet_graph()
            # dump_to_npy(images, sub=b)
            # dump_to_npy(label, sub=b)
            # dump_to_npy(loss, sub=b)
            # dump_to_npy(logits, sub=b)

            end_t = time.time()

            if b % args.loss_print_every_n_iter == 0:
                correct_of = 0.0
                predictions = predictions.to_consistent(sbp=[flow.sbp.broadcast])
                predictions = predictions.to_local()
                of_predictions = predictions.numpy()
                clsidxs = np.argmax(of_predictions, axis=1)
                label = label.to_consistent(sbp=[flow.sbp.broadcast])
                label = label.to_local()
                label_nd = label.numpy()

                for i in range(vars.total_train_batch_size):
                    if clsidxs[i] == label_nd[i]:
                        correct_of += 1

                loss = loss.to_consistent(sbp=[flow.sbp.broadcast])
                loss = loss.to_local()
                loss_np = loss.numpy()
                of_losses.append(loss_np)

                if vars.rank == 0:
                    # writer_train_loss.write("%f\n" % l)
                    # writer_train_loss.flush()
                    # writer_train_top1_acc.write("%f\n" % (correct_of / total_train_batch_size))
                    # writer_train_top1_acc.flush()
                    print(
                        "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}".format(
                            "train", epoch, b, loss_np, correct_of / vars.total_train_batch_size, -1, -1
                        )
                    )

            # if b >= 100:
            #     break

        # break

        # begin eval
        print("rank {} epoch {} train done, start validation".format(vars.rank, epoch))

        alexnet_module.eval()
        correct_of = 0.0
        num_val_samples = 0.0

        for b in range(vars.val_batches_per_epoch):
            start_t = time.time()

            predictions, label = alexnet_eval_graph()

            predictions = predictions.to_consistent(sbp=[flow.sbp.broadcast])
            predictions = predictions.to_local()

            of_predictions = predictions.numpy()

            clsidxs = np.argmax(of_predictions, axis=1)

            label = label.to_consistent(sbp=[flow.sbp.broadcast])
            label = label.to_local()

            label_nd = label.numpy()

            correct_of += (clsidxs == label_nd).sum()
            num_val_samples += label_nd.size

            end_t = time.time()

        top1 = correct_of / num_val_samples
        of_accuracy.append(top1)
        if vars.rank == 0:
            writer_val_top1_acc.write("%f\n" % (correct_of / vars.total_train_batch_size))
            writer_val_top1_acc.flush()
            print(
                "{}: epoch {}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}".format(
                    "validation", epoch, top1, -1, -1
                )
            )

        flow.save(
            alexnet_module.state_dict(),
            os.path.join(
                args.save_checkpoint_path,
                "epoch_%d_val_acc_%f" % (epoch, top1),
            ),
            consistent_dst_rank=0
        )

    # if vars.rank == 0:
    #     writer = open("graph_of_losses.txt", "w")
    #     for o in of_losses:
    #         writer.write("%f\n" % o)
    #     writer.close()

    #     writer = open("graph/accuracy.txt", "w")
    #     for o in of_accuracy:
    #         writer.write("%f\n" % o)
    #     writer.close()


if __name__ == "__main__":
    args = parse_args()
    assert _GLOBAL_VARS is None
    _GLOBAL_VARS = GlobalVars(args)
    main(args)
