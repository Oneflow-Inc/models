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
    parser = argparse.ArgumentParser(description="Inception-V3 Config", allow_abbrev=True)
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="where to save checkpoint")
    parser.add_argument("--load_path", type=str, default=None, help="where to load checkpoint")
    parser.add_argument("--ofrecord-path", type=str, default="./ofrecord", help="dataset path")
    parser.add_argument("--ofrecord-part-num", type=int, default=1, dest="ofrecord_part_num", help="ofrecord data part number")
    parser.add_argument("--use-gpu-decode", action="store_true", dest="use_gpu_decode", help="Use gpu decode")
    parser.add_argument("--synthetic-data", action="store_true", dest="synthetic_data", help="Use synthetic data")
    # training hyper-parameters
    parser.add_argument("--train-batch-size", type=int, default=32, dest="train_batch_size", help="train batch size")
    parser.add_argument("--val-batch-size", type=int, default=32, dest="val_batch_size", help="val batch size")
    parser.add_argument("--train-global-batch-size", type=int, default=None, dest="train_global_batch_size", help="train batch size")
    parser.add_argument("--val-global-batch-size", type=int, default=None, dest="val_global_batch_size", help="val batch size",)
    parser.add_argument("--num-devices-per-node", type=int, default=1, dest="num_devices_per_node", help="")
    parser.add_argument("--num-nodes", type=int, default=1, dest="num_nodes", help="node/machine number for training",)
    parser.add_argument("--lr", type=float, default=0.001, dest="learning_rate")
    parser.add_argument("--wd", type=float, default=5e-4, dest="weight_decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--num-epochs", type=int, default=90, dest="num_epochs", help="number of epochs")
    # data pre-process
    parser.add_argument("--num-classes",type=int, default=1000, dest="num_classes", help="num of pic classes",)
    parser.add_argument("--samples-per-epoch", type=int, default=9469, dest="samples_per_epoch", help="train data size")
    parser.add_argument("--val-samples-per-epoch", type=int, default=3925, dest="val_samples_per_epoch", help="validation pic number")
    parser.add_argument("--batches-per-epoch", type=int, default=None, dest="batches_per_epoch")
    parser.add_argument("--val-batches-per-epoch", type=int, default=None, dest="val_batches_per_epoch")
    parser.add_argument("--total-batches", type=int, default=-1, dest="total_batches")
    parser.add_argument("--channels-last", action="store_true", dest="channels_last")

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

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