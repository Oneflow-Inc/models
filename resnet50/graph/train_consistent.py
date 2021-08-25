import argparse
import numpy as np
import os
import time
import math

import sys
sys.path.append(".")

import oneflow as flow

from models.resnet50 import resnet50
from utils.ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
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

from oneflow.nn.module import Module
class LabelSmoothLoss(Module):
    def __init__(self, num_classes=-1, smooth_rate=0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth_rate = smooth_rate

    def forward(self, input, label):
        onehot_label = flow.F.one_hot(label, num_classes= self.num_classes, 
                                                on_value=1-self.smooth_rate, 
                                                off_value=self.smooth_rate/(self.num_classes-1))
        log_prob = input.softmax(dim=-1).log()
        onehot_label = flow.F.cast(onehot_label, log_prob.dtype)
        loss = flow.mul(log_prob * -1, onehot_label).sum(dim=-1).mean()
        return loss


def main(args):

    rank = flow.distributed.get_rank()
    world_size = flow.distributed.get_world_size()

    device_list = [i for i in range(args.process_num_per_node)]
    placement = flow.placement("cpu", {0: device_list})
    sbp = [flow.sbp.split(0)]

    total_train_batch_size = args.train_batch_size_per_device * world_size
    total_val_batch_size = args.val_batch_size_per_device * world_size
    batches_per_epoch = math.ceil(args.train_examples_num // total_train_batch_size)
    val_batches_per_epoch = int(args.val_examples_num / total_val_batch_size)
    warmup_batches = batches_per_epoch * args.warmup_epochs
    num_train_batches = batches_per_epoch * args.num_epochs
    decay_batches = num_train_batches - warmup_batches # TODO: remove warmup_batches

    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=args.train_examples_num,
        batch_size=total_train_batch_size,
        total_batch_size=total_train_batch_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="validation",
        dataset_size=args.val_examples_num,
        batch_size=total_val_batch_size,
        total_batch_size=total_val_batch_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )
    all_val_samples = len(val_data_loader) * total_val_batch_size

    # oneflow init
    start_t = time.time()
    resnet50_module = resnet50()
    
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    of_cross_entropy = flow.nn.CrossEntropyLoss()
    # of_cross_entropy = LabelSmoothLoss(num_classes=args.num_classes, smooth_rate=args.label_smoothing)

    placement = flow.placement("cuda", {0: device_list})
    sbp = [flow.sbp.broadcast]
    resnet50_module.to_consistent(placement=placement, sbp=sbp)
    of_cross_entropy.to_consistent(placement=placement, sbp=sbp)

    if args.load_checkpoint != "":
        loaded_state_dict = flow.load(
            args.load_checkpoint, consistent_src_rank=0
        )
        print("rank %d load_checkpoint >>>>>>>>> " % rank, args.load_checkpoint)
        resnet50_module.load_state_dict(loaded_state_dict)

    of_sgd = flow.optim.SGD(
        resnet50_module.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=3.05175781e-05
    )

    cosine_annealing_lr = flow.optim.lr_scheduler.CosineDecayLR(
        of_sgd, decay_steps=decay_batches
    )
    if args.warmup_epochs > 0:
        # cosine_annealing_lr = flow.optim.lr_scheduler.WarmUpLR(
        #     cosine_annealing_lr, steps=warmup_batches, start_multiplier=0
        # )
        cosine_annealing_lr = flow.optim.lr_scheduler.WarmUpLR(
            cosine_annealing_lr, warmup_factor=0, warmup_iters=warmup_batches, warmup_method="linear"
        )

    # flow.backends.nccl.boxing_fusion_threshold_mb(args.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
    # flow.backends.nccl.boxing_fusion_max_ops_num(args.nccl_fusion_max_ops)
    flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)
    if args.use_fp16 and args.num_nodes * args.process_num_per_node > 1:
        # flow.backends.nccl.boxing_fusion_all_reduce_use_buffer(False)
        flow.boxing.nccl.enable_use_buffer_to_fuse_all_reduce(False)

    class Resnet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.train_data_loader = train_data_loader
            self.resnet50 = resnet50_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer(of_sgd, lr_sch=cosine_annealing_lr)

            if args.use_fp16:
                self.config.enable_amp(True)
                # loss_scale = flow.nn.graph.amp.DynamicLossScalePolicy(increment_period=2000)
                # self.config.amp_add_loss_scale_policy(loss_scale)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)

            # self.config.enable_fuse_add_to_output(True)
            self.config.allow_fuse_add_to_output(True)
            # self.config.prune_parallel_cast_ops(True)
            self.config.proto.set_prune_parallel_cast_ops(True)
            # self.config.enable_inplace(True)
            self.config.proto.set_enable_inplace(True)
            if args.num_nodes > 1:
                # self.config.cudnn_conv_heuristic_search_algo(True)
                self.config.proto.set_cudnn_conv_heuristic_search_algo(True)
            else:
                # self.config.cudnn_conv_heuristic_search_algo(False)
                self.config.proto.set_cudnn_conv_heuristic_search_algo(False)

            # self.config.enable_fuse_model_update_ops(True)
            self.config.allow_fuse_model_update_ops(True)
        
        def build(self):
            image, label = self.train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            logits = self.resnet50(image)
            loss = self.cross_entropy(logits, label)
            predictions = logits.softmax()
            loss.backward()
            return loss, predictions, label

    resnet50_graph = Resnet50Graph()

    class Resnet50EvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.val_data_loader = val_data_loader
            self.resnet50 = resnet50_module
            
            if args.use_fp16:
                self.config.enable_amp(True)

            # self.config.enable_fuse_add_to_output(True)
            self.config.allow_fuse_add_to_output(True)
        
        def build(self):
            image, label = self.val_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            with flow.no_grad():
                logits = self.resnet50(image)
                predictions = logits.softmax()
            return predictions, label

    resnet50_eval_graph = Resnet50EvalGraph()

    if rank == 0:
        writer_train_loss = open("graph/train_losses.txt", "w")
        writer_train_top1_acc = open("graph/train_top1_accuracy.txt", "w")
        writer_val_top1_acc = open("graph/val_top1_accuracy.txt", "w")

    of_losses, of_accuracy = [], []
    for epoch in range(args.num_epochs):
        resnet50_module.train()

        for b in range(batches_per_epoch):
            # oneflow graph train
            start_t = time.time()

            loss, predictions, label = resnet50_graph()

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
            
                for i in range(total_train_batch_size):
                    if clsidxs[i] == label_nd[i]:
                        correct_of += 1
                loss = loss.to_consistent(sbp=[flow.sbp.broadcast])
                loss = loss.to_local()
                l = loss.numpy()
                of_losses.append(l)
                if rank == 0:
                    # writer_train_loss.write("%f\n" % l)
                    # writer_train_loss.flush()
                    # writer_train_top1_acc.write("%f\n" % (correct_of / total_train_batch_size))
                    # writer_train_top1_acc.flush()
                    print(
                        "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}".format(
                            "train", epoch, b, l, correct_of / total_train_batch_size, -1, -1
                        )
                    )

        resnet50_module.eval()
        correct_of = 0.0
        for b in range(val_batches_per_epoch):
            start_t = time.time()
            predictions, label = resnet50_eval_graph()
            
            predictions = predictions.to_consistent(sbp=[flow.sbp.broadcast])
            predictions = predictions.to_local()

            of_predictions = predictions.numpy()
            
            clsidxs = np.argmax(of_predictions, axis=1)

            label = label.to_consistent(sbp=[flow.sbp.broadcast])
            label = label.to_local()

            label_nd = label.numpy()
            
            for i in range(total_val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()

        top1 = correct_of / all_val_samples
        of_accuracy.append(top1)
        if rank == 0:
            writer_val_top1_acc.write("%f\n" % (correct_of / total_train_batch_size))
            writer_val_top1_acc.flush()
            print(
                "{}: epoch {}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}".format(
                    "validation", epoch, top1, -1, -1 
                )
            )

        flow.save(
            resnet50_module.state_dict(),
            os.path.join(
                args.save_checkpoint_path,
                "epoch_%d_val_acc_%f" % (epoch, top1),
            ),
            consistent_dst_rank=0
        )

    # if rank == 0:
    #     writer = open("graph_of_losses.txt", "w")
    #     for o in of_losses:
    #         writer.write("%f\n" % o)
    #     writer.close()

    #     writer = open("graph/accuracy.txt", "w")
    #     for o in of_accuracy:
    #         writer.write("%f\n" % o)
    #     writer.close()

if __name__ == "__main__":
    args = _parse_args()
    main(args)
