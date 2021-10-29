import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import argparse
import numpy as np
import time

import oneflow as flow

from models.resnet50 import resnet50
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test resnet50")
    parser.add_argument(
        "--model",
        type=str,
        default="./resnet50_imagenet_pretrain_model",
        dest="model_path",
        help="model path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        required=True,
        dest="image_path",
        help="input image path",
    )
    parser.add_argument("--graph", action="store_true", help="Run model in graph mode.")
    return parser.parse_args()


class InferGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, image):
        with flow.no_grad():
            logits = self.model(image)
            pred = logits.softmax()
        return pred


def main(args):
    start_t = time.perf_counter()

    print("***** Model Init *****")
    model = resnet50()
    model.load_state_dict(flow.load(args.model_path))
    model = model.to("cuda")
    model.eval()
    end_t = time.perf_counter()
    print(f"***** Model Init Finish, time escapled {end_t - start_t:.6f} s *****")

    if args.graph:
        model_graph = InferGraph(model)

    start_t = end_t
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    if args.graph:
        pred = model_graph(image)
    else:
        pred = model(image).softmax()

    pred = pred.numpy()
    prob = np.max(pred)
    clsidx = np.argmax(pred)
    cls = clsidx_2_labels[clsidx]

    end_t = time.perf_counter()
    print(
        "predict image ({}) prob: {:.5f}, class name: {}, time escapled: {:.6f} s".format(
            os.path.basename(args.image_path), prob, cls, end_t - start_t
        )
    )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
