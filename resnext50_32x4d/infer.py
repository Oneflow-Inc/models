import oneflow as flow

import argparse
import numpy as np
import time

from models.resnext50_32x4d import resnext50_32x4d
from utils.imagenet1000_clsidx_to_labels import clsidx_2_labels
from utils.numpy_data_utils import load_image


def _parse_args():
    parser = argparse.ArgumentParser("flags for test resnext50_32x4d")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./resnext50_32x4d_oneflow_model",
        help="model path",
    )
    parser.add_argument("--image_path", type=str, default="", help="input image path")
    return parser.parse_args()


def main(args):
    
    

    start_t = time.time()
    resnext50_32x4d_module = resnext50_32x4d()
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    start_t = time.time()
    pretrain_models = flow.load(args.model_path)
    resnext50_32x4d_module.load_state_dict(pretrain_models)
    end_t = time.time()
    print("load params time : {}".format(end_t - start_t))

    resnext50_32x4d_module.eval()
    resnext50_32x4d_module.to("cuda")

    start_t = time.time()
    image = load_image(args.image_path)
    image = flow.Tensor(image, device=flow.device("cuda"))
    predictions = resnext50_32x4d_module(image).softmax()
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
