import oneflow as flow

import argparse
import numpy as np
import time

from models.mobilenetv3 import mobilenet_v3_small
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test mobilenetv3")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./mobilenetv3_oneflow_model",
        help="model path",
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    return parser.parse_args()


def main(args):

    start_t = time.time()
    mobilenetv3_module = mobilenet_v3_small()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    mobilenetv3_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    mobilenetv3_module.eval()
    mobilenetv3_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = mobilenetv3_module(image).softmax()
    predictions = predictions.numpy()
    end_t = time.time()
    print("infer time : {}".format(end_t - start_t))
    clsidx = np.argmax(predictions)
    print(
        "predict prob: %f, class name: %s"
        % (np.max(predictions), clsidx_2_labels[clsidx])
    )


if __name__ == "__main__":
    args = _parse_args()
    main(args)
