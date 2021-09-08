from numpy import random
from pytorch_train import train_pytorch
from oneflow_train import train_oneflow
import argparse
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser("flags for time testing")
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--warmup_iters", type=int, default=5, help="warmup iters")
    parser.add_argument("--bp_iters", type=int, default=95, help="bp iters")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for speed testing"
    )
    parser.add_argument(
        "--test_code", type=str, choices=['pytorch', 'oneflow'], default="oneflow", help="choose the code for testing"
    )
    return parser.parse_args()

def main(args):
    batch_size = args.batch_size
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)
    if args.test_code == "pytorch":
        train_pytorch(args, image_nd, label_nd)
    elif args.test_code == "oneflow":
        train_oneflow(args, image_nd, label_nd)

if __name__ == "__main__":
    args = _parse_args()
    main(args)