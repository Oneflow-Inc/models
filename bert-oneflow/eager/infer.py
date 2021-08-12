import argparse
import time

import numpy as np
import oneflow as flow
from model.bert import BERT


def _parse_args():
    parser = argparse.ArgumentParser("flags for test bert")
    parser.add_argument(
        "--model-path", type=str, metavar="DIR", help="model path"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="model device"
    )
    parser.add_argument(
        "-hs", "--hidden", type=int, default=256, 
        help="hidden size of transformer model"
    )
    parser.add_argument(
        "--layers", type=int, default=8, 
        help="number of layers in transfomer model"
    )
    parser.add_argument(
        "--attn-heads", type=int, default=8, 
        help="number of attention heads"
    )
    parser.add_argument("--input_path", type=str, default="",
                        help="input string for prediction")
    return parser.parse_args()


def inference(args):
    start_t = time.time()
    bert_module = BERT(
        244514, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    end_t = time.time()
    print("Initialize model using time: {:.3f}s".format(end_t - start_t))

    start_t = time.time()
    bert_module.load_state_dict(flow.load(args.model_path))
    end_t = time.time()
    print("Loading parameters using time: {:.3f}s".format(end_t - start_t))

    bert_module.eval()
    bert_module.to(args.device)

    start_t = time.time()
    inputs = [0, 1, 2]
    inputs = flow.Tensor(inputs, device=flow.device(args.device))
    prediction = bert_module(inputs).numpy()
    end_t = time.time()
    print("Inference using time: {:.3f}".format(end_t - start_t))


if __name__ == "__main__":
    args = _parse_args()
    inference(args)
