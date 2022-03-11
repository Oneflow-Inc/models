import argparse
import math
import oneflow as flow

_GLOBAL_ARGS = None


def get_args():
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args()

    return _GLOBAL_ARGS


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args(ignore_unknown_args=False):
    parser = argparse.ArgumentParser(
        description="OneFlow ResNet50 Arguments", allow_abbrev=False
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        dest="save_path",
        help="root dir of saving checkpoint",
    )
    parser.add_argument(
        "--save-init",
        action="store_true",
        dest="save_init",
        help="save right on init model finished",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        dest="load_path",
        help="root dir of loading checkpoint",
    )
    parser.add_argument(
        "--ofrecord-path",
        type=str,
        default="./ofrecord",
        dest="ofrecord_path",
        help="dataset path",
    )
    parser.add_argument(
        "--ofrecord-part-num",
        type=int,
        default=1,
        dest="ofrecord_part_num",
        help="ofrecord data part number",
    )
    parser.add_argument(
        "--use-gpu-decode",
        action="store_true",
        dest="use_gpu_decode",
        help="Use gpu decode.",
    )
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        dest="synthetic_data",
        help="Use synthetic data",
    )

    # fuse bn relu or bn add relu
    parser.add_argument(
        "--fuse-bn-relu",
        action="store_true",
        dest="fuse_bn_relu",
        help="Whether to use use fuse batch_normalization and relu.",
    )
    parser.add_argument(
        "--fuse-bn-add-relu",
        action="store_true",
        dest="fuse_bn_add_relu",
        help="Whether to use use fuse batch_normalization, add and relu.",
    )

    # training hyper-parameters
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=32,
        dest="train_batch_size",
        help="train batch size",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=32,
        dest="val_batch_size",
        help="val batch size",
    )
    parser.add_argument(
        "--train-global-batch-size",
        type=int,
        default=None,
        dest="train_global_batch_size",
        help="train batch size",
    )
    parser.add_argument(
        "--val-global-batch-size",
        type=int,
        default=None,
        dest="val_global_batch_size",
        help="val batch size",
    )
    parser.add_argument(
        "--num-devices-per-node",
        type=int,
        default=1,
        dest="num_devices_per_node",
        help="",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        dest="num_nodes",
        help="node/machine number for training",
    )
    parser.add_argument("--lr", type=float, default=0.256, dest="learning_rate")
    parser.add_argument("--wd", type=float, default=1.0 / 32768, dest="weight_decay")
    parser.add_argument("--momentum", type=float, default=0.875, help="momentum")
    parser.add_argument(
        "--lr-decay-type",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        dest="lr_decay_type",
        help="cosine, step",
    )
    parser.add_argument(
        "--grad-clipping",
        type=float,
        default=0.0,
        dest="grad_clipping",
        help="gradient clipping",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        dest="warmup_epochs",
        help="the epochs to warmp-up lr to scaled large-batch value",
    )
    parser.add_argument("--legacy-init", action="store_true", dest="legacy_init")
    parser.add_argument(
        "--use-fp16", action="store_true", help="Run model in fp16 mode."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=90, dest="num_epochs", help="number of epochs"
    )
    parser.add_argument(
        "--nccl-fusion-threshold-mb",
        type=int,
        default=16,
        dest="nccl_fusion_threshold_mb",
        help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--nccl-fusion-max-ops",
        type=int,
        default=24,
        dest="nccl_fusion_max_ops",
        help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--zero-init-residual",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        dest="zero_init_residual",
    )
    parser.add_argument(
        "--scale-grad",
        action="store_true",
        dest="scale_grad",
        help="scale init grad with world_size",
    )

    # for data process
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        dest="num_classes",
        help="num of pic classes",
    )
    parser.add_argument(
        "--channel-last", action="store_true", dest="channel_last",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=1281167,
        dest="samples_per_epoch",
        help="train pic number",
    )
    parser.add_argument(
        "--val-samples-per-epoch",
        type=int,
        default=50000,
        dest="val_samples_per_epoch",
        help="validation pic number",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        dest="label_smoothing",
        help="label smoothing factor",
    )
    parser.add_argument(
        "--batches-per-epoch", type=int, default=None, dest="batches_per_epoch",
    )
    parser.add_argument(
        "--val-batches-per-epoch", type=int, default=None, dest="val_batches_per_epoch",
    )
    parser.add_argument(
        "--total-batches", type=int, default=-1, dest="total_batches",
    )
    parser.add_argument("--skip-eval", action="store_true", dest="skip_eval")

    # log and loss print
    parser.add_argument(
        "--print-interval",
        type=int,
        default=100,
        dest="print_interval",
        help="print loss every n iteration",
    )
    parser.add_argument(
        "--print-timestamp", action="store_true", dest="print_timestamp",
    )
    parser.add_argument(
        "--metric-local",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        dest="metric_local",
    )
    parser.add_argument(
        "--metric-train-acc",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        dest="metric_train_acc",
    )
    parser.add_argument(
        "--gpu-stat-file",
        type=str,
        default=None,
        dest="gpu_stat_file",
        help="stat gpu utilization and memory usage when print",
    )

    parser.add_argument("--graph", action="store_true", help="Run model in graph mode.")
    parser.add_argument("--ddp", action="store_true", help="Run model in ddp mode.")

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if args.num_nodes > 1:
        raise ValueError("NOT support num_nodes > 1")

    if args.ddp and args.graph:
        raise ValueError("graph and ddp can't be set at the same time")

    if args.use_fp16 and not args.graph:
        raise ValueError("NOT support fp16 in eager mode")

    if args.ddp and not args.metric_local:
        raise ValueError("metric_local must be set to True when with ddp")

    if args.ddp and args.scale_grad:
        raise ValueError("scale_grad is unavailable with ddp")

    world_size = flow.env.get_world_size()
    if args.train_global_batch_size is None:
        args.train_global_batch_size = args.train_batch_size * world_size
    else:
        assert args.train_global_batch_size % args.train_batch_size == 0

    if args.val_global_batch_size is None:
        args.val_global_batch_size = args.val_batch_size * world_size
    else:
        assert args.val_global_batch_size % args.val_batch_size == 0

    if args.batches_per_epoch is None:
        args.batches_per_epoch = math.ceil(
            args.samples_per_epoch // args.train_global_batch_size
        )

    if args.val_batches_per_epoch is None:
        args.val_batches_per_epoch = int(
            args.val_samples_per_epoch / args.val_global_batch_size
        )

    if flow.env.get_rank() == 0:
        _print_args(args)

    return args


def _print_args(args):
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


if __name__ == "__main__":
    get_args()
